"""保存本轮原始数据；逐字节相同的历史CSV引用旧归档，不重复存储。"""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path.cwd()
AUDIT = ROOT.parent.parent / "audit/fuzzy-validation-2026-10-28"
OUT = ROOT / "docs/evidence/fuzzy-validation-2026-10-28"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


known = {}
old_c1 = OUT.parent / "c1-sensor-study-2026-10-25"
old_manifest = read(old_c1 / "manifest.json")
for member, record in old_manifest["raw_members"].items():
    if member.endswith(".csv"):
        known[record["sha256"]] = (old_c1 / record["archive"], member)
old_m1 = OUT.parent / "fuzzy-pi-2026-10-25/raw_data.zip"
with zipfile.ZipFile(old_m1) as z:
    for member, digest in json.loads(z.read("SHA256.json")).items():
        if member.endswith(".csv"):
            known[digest] = (old_m1, member)

files = []
for kind in ["M", "C"]:
    source = ROOT / f"results/V20261028/{kind}/run_001"
    for name in ["comparison.csv", "study_summary.json", "study_config.json", "online_gains.png", "force_errors.png", "combo_statistics.csv"]:
        if (source / name).exists():
            shutil.copy2(source / name, OUT / (kind.lower()+"_"+name))
    for file in sorted(source.rglob("*")):
        if file.is_file() and "_中文" not in file.name and (
                file.name in {"contact_force.csv", "control.csv", "joint_state.csv", "raw_contacts.csv", "c1_curves.png"}
                or file.suffix in {".json", ".xml", ".log"}):
            files.append((kind+"/"+file.relative_to(source).as_posix(), file))
for file in sorted((ROOT / "results/V20261028/t").rglob("*")):
    if file.is_file() and "_中文" not in file.name:
        files.append(("debug_initial_model_load/"+file.relative_to(ROOT / "results/V20261028/t").as_posix(), file))
for file in sorted((ROOT / "t_v28").rglob("*")):
    if file.is_file() and "_中文" not in file.name and file.suffix in {".csv", ".json", ".xml", ".log"}:
        files.append(("short_path_physical_test/"+file.relative_to(ROOT / "t_v28").as_posix(), file))
for file in sorted(AUDIT.iterdir()):
    if file.suffix in {".py", ".log"}:
        files.append(("provenance/"+file.name, file))
for folder in ["common", "controllers", "experiments", "tests", "configs"]:
    for file in sorted((ROOT / folder).glob("*")):
        if file.is_file() and file.suffix in {".py", ".json"}:
            files.append(("source/"+file.relative_to(ROOT).as_posix(), file))
for name in ["verify_results.py", "package_results.py"]:
    shutil.copy2(AUDIT / name, OUT / name)

members, archives, pending, reused = {}, {}, [], 0
for name, file in files:
    digest = sha(file)
    if digest in known and file.suffix == ".csv":
        archive, old_member = known[digest]
        members[name] = {"archive": "../"+archive.parent.name+"/"+archive.name,
                         "member": old_member, "sha256": digest, "bytes": file.stat().st_size}
        reused += 1
    else:
        pending.append((name, file, digest))
chunks, chunk, size = [], [], 0
for record in pending:
    if chunk and size + record[1].stat().st_size > 80_000_000:
        chunks.append(chunk)
        chunk, size = [], 0
    chunk.append(record)
    size += record[1].stat().st_size
if chunk:
    chunks.append(chunk)
for index, chunk in enumerate(chunks, 1):
    archive = OUT / f"raw_data_{index:02d}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, file, digest in chunk:
            z.write(file, name)
            members[name] = {"archive": archive.name, "member": name,
                             "sha256": digest, "bytes": file.stat().st_size}
    print(f"已保存 {archive.name}：{archive.stat().st_size/1e6:.1f} MB", flush=True)

# 连同引用的历史文件验证每个成员，避免只校验ZIP目录或文件名。
by_archive = {}
for name, record in members.items():
    by_archive.setdefault(record["archive"], []).append((name, record))
for relative, records in by_archive.items():
    archive = OUT / relative
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None, relative
        for name, record in records:
            content = z.read(record["member"])
            assert len(content) == record["bytes"]
            assert hashlib.sha256(content).hexdigest() == record["sha256"], name
    archives[relative] = {"sha256": sha(archive), "bytes": archive.stat().st_size,
                          "referenced_members": len(records), "historical": relative.startswith("../")}
manifest = {"date": "2026-10-28", "raw_members": members, "archives": archives,
            "historical_csv_references": reused,
            "extraction": "按raw_members逐项从archive读取member，校验SHA256后写到逻辑路径。../表示引用已有历史归档；不要仅解压本目录的新ZIP。"}
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(json.dumps({"members": len(members), "historical_csv_references": reused,
                  "new_archives": len(chunks), "new_archive_bytes": sum(v["bytes"] for v in archives.values() if not v["historical"]),
                  "archive_content_sha256_and_crc": "PASS"}, ensure_ascii=False, indent=2))
