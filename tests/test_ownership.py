"""检查归属（原 Study UID -> 原 PatientID）不变量测试。

覆盖：
- 同批冲突整批拒绝；跨批/并发冲突只有归属一致的批次可发布；
- 并发首传冲突至多一批成功；
- 失败/中断批次的实例与映射不占用归属；
- 归属只取实例顶层字段，SR 嵌套引用外部 Study 不代表拥有；
- 同一患者后续序列、同字节重传保持可用。
"""
from __future__ import annotations

import io
import multiprocessing as mp
import os
import sys
import threading
import zipfile
from pathlib import Path

import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, EnhancedSRStorage, ExplicitVRLittleEndian
from werkzeug.datastructures import MultiDict

from samples import generate_samples as gen


# --------------------------------------------------------------------- 辅助
def _make_ct(path: Path, patient_id: str, study_uid: str, series_uid: str,
             sop_uid: str, pixel_seed: int = 1) -> None:
    """生成一个最小可用、未压缩 Explicit VR LE 的 CT DICOM。"""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), Dataset(), file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.PatientName = "CASE^TEST"
    ds.PatientID = patient_id
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = sop_uid
    ds.Modality = "CT"
    ds.Rows, ds.Columns = 8, 8
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    import numpy as np
    rng = np.random.default_rng(pixel_seed)
    ds.PixelData = rng.integers(0, 1000, size=(8, 8), dtype="<u2").tobytes()
    ds.save_as(str(path), write_like_original=False)


def _make_sr_with_external_ref(path: Path, patient_id: str, own_study_uid: str,
                               own_series_uid: str, sop_uid: str,
                               external_study_uid: str) -> None:
    """SR：顶层属于 own_study_uid，嵌套引用另一个外部 Study。"""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = EnhancedSRStorage
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), Dataset(), file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.PatientName = "CASE^SR"
    ds.PatientID = patient_id
    ds.StudyInstanceUID = own_study_uid
    ds.SeriesInstanceUID = own_series_uid
    ds.SOPClassUID = EnhancedSRStorage
    ds.SOPInstanceUID = sop_uid
    ds.Modality = "SR"
    ref = Dataset()
    ref.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.1"
    ref.ReferencedSOPInstanceUID = external_study_uid
    ds.ReferencedStudySequence = [ref]
    ds.save_as(str(path), write_like_original=False)


def _upload(client, paths):
    data = MultiDict()
    handles = []
    for p in paths:
        fh = open(p, "rb")
        handles.append(fh)
        data.add("dicom_files", (fh, Path(p).name, "application/dicom"))
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    for fh in handles:
        fh.close()
    return resp


def _batch_id(resp) -> str:
    assert resp.status_code == 302
    return resp.headers["Location"].rstrip("/").split("/")[-1]


def _download_status(client, batch_id: str) -> int:
    return client.get(f"/batches/{batch_id}/download").status_code


def _is_failed(client, batch_id: str) -> bool:
    return b"failed" in client.get(f"/batches/{batch_id}").data


STUDY_X = "1.2.3.4.5.6.7.8.9.1"
STUDY_Y = "1.2.3.4.5.6.7.8.9.2"
# 从未作为任何实例顶层 Study 上传过的外部 UID，仅被 SR 嵌套引用
STUDY_EXTERNAL = "1.2.3.4.5.6.7.8.9.99"
SERIES_A = "1.2.3.4.5.6.7.8.10.1"
SERIES_B = "1.2.3.4.5.6.7.8.10.2"
PID_A = "PID-AAA"
PID_B = "PID-BBB"


# ----------------------------------------------------- 1. 同批冲突整批拒绝
def test_same_batch_study_with_two_patients_rejected(client, tmp_path):
    d = tmp_path / "batch"
    d.mkdir()
    f1 = d / "a.dcm"
    f2 = d / "b.dcm"
    _make_ct(f1, PID_A, STUDY_X, SERIES_A, "1.2.3.4.5.6.7.8.20.1", 1)
    _make_ct(f2, PID_B, STUDY_X, SERIES_B, "1.2.3.4.5.6.7.8.20.2", 2)

    bid = _batch_id(_upload(client, [f1, f2]))
    assert _is_failed(client, bid)
    assert _download_status(client, bid) == 409
    detail = client.get(f"/batches/{bid}").data.decode()
    assert "StudyInstanceUID" in detail and "PatientID" in detail


# ----------------------------------------------------- 2. 跨批冲突只拒后者
def test_cross_batch_ownership_conflict_rejects_second(client, tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    f_a = d / "patient_a.dcm"
    f_b = d / "patient_b.dcm"
    _make_ct(f_a, PID_A, STUDY_X, SERIES_A, "1.2.3.4.5.6.7.8.20.1", 1)
    _make_ct(f_b, PID_B, STUDY_X, SERIES_B, "1.2.3.4.5.6.7.8.20.2", 2)

    bid1 = _batch_id(_upload(client, [f_a]))
    assert _download_status(client, bid1) == 200

    bid2 = _batch_id(_upload(client, [f_b]))
    assert _is_failed(client, bid2)
    assert _download_status(client, bid2) == 409

    # 既有批次不受影响，仍可下载
    assert _download_status(client, bid1) == 200

    # 归属一致（同一患者、同 Study 的新序列/实例）应继续成功
    f_a2 = d / "patient_a_more.dcm"
    _make_ct(f_a2, PID_A, STUDY_X, SERIES_B, "1.2.3.4.5.6.7.8.20.3", 3)
    bid3 = _batch_id(_upload(client, [f_a2]))
    assert _download_status(client, bid3) == 200


# ----------------------------------------------------- 3. 同字节重传可用
def test_identical_reupload_same_owner_kept_available(client, tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    f = d / "a.dcm"
    _make_ct(f, PID_A, STUDY_X, SERIES_A, "1.2.3.4.5.6.7.8.20.1", 1)

    bid1 = _batch_id(_upload(client, [f]))
    assert _download_status(client, bid1) == 200
    bid2 = _batch_id(_upload(client, [f]))  # 同字节重传
    assert _download_status(client, bid2) == 200


# ----------------------------------------------------- 4. 不同 Study 互不影响
def test_different_studies_different_patients_both_publish(client, tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    f_a = d / "a.dcm"
    f_b = d / "b.dcm"
    _make_ct(f_a, PID_A, STUDY_X, SERIES_A, "1.2.3.4.5.6.7.8.20.1", 1)
    _make_ct(f_b, PID_B, STUDY_Y, SERIES_B, "1.2.3.4.5.6.7.8.20.2", 2)

    bid1 = _batch_id(_upload(client, [f_a]))
    bid2 = _batch_id(_upload(client, [f_b]))
    assert _download_status(client, bid1) == 200
    assert _download_status(client, bid2) == 200


# --------------------------------------- 5. 失败批次不占用归属，可再上传
def test_failed_batch_does_not_hold_ownership(client, tmp_path):
    from app.db import get_db

    d = tmp_path / "imgs"
    d.mkdir()
    good = d / "good.dcm"
    bad = d / "not_dicom.dcm"
    _make_ct(good, PID_A, STUDY_X, SERIES_A, "1.2.3.4.5.6.7.8.20.1", 1)
    bad.write_bytes(b"this is not dicom" * 20)

    # 含坏文件：整批失败
    bid_fail = _batch_id(_upload(client, [good, bad]))
    assert _download_status(client, bid_fail) == 409

    conn = get_db()
    try:
        # 失败批次没有留下任何归属记录
        rows = conn.execute(
            "SELECT * FROM study_ownership WHERE original_study_uid=?",
            (STUDY_X,),
        ).fetchall()
        assert rows == []
    finally:
        conn.close()

    # 随后上传未与已发布归属冲突的影像仍成功（同患者）
    bid_ok = _batch_id(_upload(client, [good]))
    assert _download_status(client, bid_ok) == 200

    # 已发布归属后，换患者再传同一 Study 仍被拒绝
    other = d / "other.dcm"
    _make_ct(other, PID_B, STUDY_X, SERIES_B, "1.2.3.4.5.6.7.8.20.9", 9)
    bid_conflict = _batch_id(_upload(client, [other]))
    assert _download_status(client, bid_conflict) == 409


# ------------------------------------------ 6. 中断批次不占用归属
def test_interrupted_batch_does_not_hold_ownership(client, sample_files,
                                                   monkeypatch, tmp_path):
    from app import services
    from app.db import get_db, recover_interrupted_batches

    boom = RuntimeError("模拟断电")
    monkeypatch.setattr(services, "_build_zip",
                        lambda batch_id, plan: (_ for _ in ()).throw(boom))

    paths = sorted(sample_files.glob("*.dcm"))
    bid = _batch_id(_upload(client, paths))
    assert _download_status(client, bid) == 409

    recover_interrupted_batches()
    assert services.get_batch(bid)["status"] == "failed"

    conn = get_db()
    try:
        # 中断批次写过实例/映射，但没有归属记录
        rows = conn.execute(
            "SELECT * FROM study_ownership WHERE original_study_uid=?",
            (gen.STUDY_UID,),
        ).fetchall()
        assert rows == []
    finally:
        conn.close()

    # 重试（同患者）成功并确立归属
    monkeypatch.undo()
    bid2 = _batch_id(_upload(client, paths))
    assert _download_status(client, bid2) == 200
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT original_patient_id FROM study_ownership "
            "WHERE original_study_uid=?",
            (gen.STUDY_UID,),
        ).fetchone()
        assert row is not None and row["original_patient_id"] == gen.PATIENT_ID
    finally:
        conn.close()


# ----------------------------- 7. SR 嵌套引用外部 Study 不构成拥有
def test_sr_nested_reference_does_not_claim_ownership(client, tmp_path):
    from app.db import get_db

    d = tmp_path / "imgs"
    d.mkdir()
    # 患者 A 拥有 STUDY_X
    ct_a = d / "ct_a.dcm"
    _make_ct(ct_a, PID_A, STUDY_X, SERIES_A, "1.2.3.4.5.6.7.8.20.1", 1)
    bid1 = _batch_id(_upload(client, [ct_a]))
    assert _download_status(client, bid1) == 200

    # 患者 B 的 SR：顶层 Study 是 STUDY_Y（自己的），仅在嵌套序列引用
    # 一个从未作为顶层 Study 上传过的外部 UID
    sr_b = d / "sr_b.dcm"
    _make_sr_with_external_ref(
        sr_b, PID_B, own_study_uid=STUDY_Y,
        own_series_uid="1.2.3.4.5.6.7.8.10.9",
        sop_uid="1.2.3.4.5.6.7.8.30.1",
        external_study_uid=STUDY_EXTERNAL,
    )
    bid2 = _batch_id(_upload(client, [sr_b]))
    assert _download_status(client, bid2) == 200

    conn = get_db()
    try:
        owners = {
            r["original_study_uid"]: r["original_patient_id"]
            for r in conn.execute("SELECT * FROM study_ownership").fetchall()
        }
        # STUDY_X 仍只归属 A；SR 顶层 STUDY_Y 归属 B；外部引用未被占有
        assert owners[STUDY_X] == PID_A
        assert owners[STUDY_Y] == PID_B
        assert STUDY_EXTERNAL not in owners
    finally:
        conn.close()

    # 反例：若患者 B 的实例顶层就声明 STUDY_X，必须拒绝
    ct_b = d / "ct_b.dcm"
    _make_ct(ct_b, PID_B, STUDY_X, SERIES_B, "1.2.3.4.5.6.7.8.20.7", 7)
    bid3 = _batch_id(_upload(client, [ct_b]))
    assert _download_status(client, bid3) == 409


# --------------------------------------- 8. 线程并发首传冲突至多一批成功
def test_concurrent_first_publish_conflict_only_one_succeeds(client, tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    f_a = d / "a.dcm"
    f_b = d / "b.dcm"
    _make_ct(f_a, PID_A, STUDY_X, SERIES_A, "1.2.3.4.5.6.7.8.20.1", 1)
    _make_ct(f_b, PID_B, STUDY_X, SERIES_B, "1.2.3.4.5.6.7.8.20.2", 2)

    outcomes: list[tuple[str, int]] = []
    lock = threading.Lock()

    def worker(tag, path):
        bid = _batch_id(_upload(client, [path]))
        with lock:
            outcomes.append((tag, _download_status(client, bid)))

    threads = [
        threading.Thread(target=worker, args=("A", f_a)),
        threading.Thread(target=worker, args=("B", f_b)),
        threading.Thread(target=worker, args=("A2", f_a)),
        threading.Thread(target=worker, args=("B2", f_b)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    statuses = dict(outcomes)
    ready = [tag for tag, code in statuses.items() if code == 200]
    failed = [tag for tag, code in statuses.items() if code == 409]
    # 成功的全部来自同一归属方，冲突方一个都不能成功
    assert ready, "至少应有一批成功"
    winner_owner = PID_A if ready[0].startswith("A") else PID_B
    if winner_owner == PID_A:
        assert set(ready) <= {"A", "A2"}
        assert set(failed) == {"B", "B2"}
    else:
        assert set(ready) <= {"B", "B2"}
        assert set(failed) == {"A", "A2"}


# ----------------------------- 9. 跨进程并发首传冲突至多一批成功
def _ownership_worker(data_dir, uid_prefix, work_dir, tag, patient_id,
                      sop_uid, queue) -> None:
    os.environ["DICOM_DATA_DIR"] = data_dir
    os.environ["DICOM_UID_PREFIX"] = uid_prefix
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from werkzeug.datastructures import MultiDict  # noqa: E402
    from app.web import create_app  # noqa: E402

    test_mod = sys.modules[__name__]
    path = Path(work_dir) / f"{tag}.dcm"
    test_mod._make_ct(path, patient_id, STUDY_X, SERIES_A, sop_uid,
                      pixel_seed=abs(hash(tag)) % 1000)
    client = create_app().test_client()
    data = MultiDict()
    data.add("dicom_files", (open(path, "rb"), path.name, "application/dicom"))
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    bid = resp.headers["Location"].rstrip("/").split("/")[-1]
    code = client.get(f"/batches/{bid}/download").status_code
    queue.put((tag, patient_id, "ready" if code == 200 else "failed"))


def test_cross_process_first_publish_conflict(app, tmp_path):
    data_dir = os.environ["DICOM_DATA_DIR"]
    uid_prefix = os.environ["DICOM_UID_PREFIX"]
    work_dir = tmp_path / "procs"
    work_dir.mkdir()
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=_ownership_worker, args=(
            data_dir, uid_prefix, str(work_dir),
            "A1", PID_A, "1.2.3.4.5.6.7.8.20.1", queue)),
        ctx.Process(target=_ownership_worker, args=(
            data_dir, uid_prefix, str(work_dir),
            "B1", PID_B, "1.2.3.4.5.6.7.8.20.2", queue)),
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0
    results = [queue.get() for _ in procs]

    ready_owners = {pid for _, pid, st in results if st == "ready"}
    assert len(ready_owners) == 1, f"冲突双方不应同时成功: {results}"
    # 归属表只落一个患者
    from app.db import get_db
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT DISTINCT original_patient_id FROM study_ownership "
            "WHERE original_study_uid=?",
            (STUDY_X,),
        ).fetchall()
        assert [r["original_patient_id"] for r in rows] == list(ready_owners)
    finally:
        conn.close()


# ----------------------------- 10. 同患者并发重传全部成功、归属唯一
def test_cross_process_same_owner_concurrent(app, tmp_path):
    data_dir = os.environ["DICOM_DATA_DIR"]
    uid_prefix = os.environ["DICOM_UID_PREFIX"]
    work_dir = tmp_path / "procs2"
    work_dir.mkdir()
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=_ownership_worker, args=(
            data_dir, uid_prefix, str(work_dir),
            f"A{i}", PID_A, f"1.2.3.4.5.6.7.8.20.{10 + i}", queue))
        for i in range(3)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0
    results = [queue.get() for _ in procs]
    assert all(st == "ready" for _, _, st in results), results

    from app.db import get_db
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT DISTINCT original_patient_id FROM study_ownership "
            "WHERE original_study_uid=?",
            (STUDY_X,),
        ).fetchall()
        assert [r["original_patient_id"] for r in rows] == [PID_A]
    finally:
        conn.close()


# ----------------------------- 11. 发布后的 zip/manifest 与归属一致
def test_manifest_consistent_with_ownership(client, tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    f_a1 = d / "a1.dcm"
    f_a2 = d / "a2.dcm"
    _make_ct(f_a1, PID_A, STUDY_X, SERIES_A, "1.2.3.4.5.6.7.8.20.1", 1)
    _make_ct(f_a2, PID_A, STUDY_X, SERIES_B, "1.2.3.4.5.6.7.8.20.2", 2)
    bid = _batch_id(_upload(client, [f_a1, f_a2]))
    resp = client.get(f"/batches/{bid}/download")
    assert resp.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(resp.data))
    manifest = zf.read("manifest.csv").decode("utf-8-sig").splitlines()[1:]
    # 清单中该 Study 只出现一个原 PatientID
    study_pid_pairs = {
        (line.split(",")[3], line.split(",")[1]) for line in manifest
    }
    assert study_pid_pairs == {(STUDY_X, PID_A)}
