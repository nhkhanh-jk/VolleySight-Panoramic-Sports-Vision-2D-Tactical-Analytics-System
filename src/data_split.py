"""Phase 1.1 - Audit dataset va tao split khong ro ri (leak) theo video_id.

Chay: python3 src/data_split.py
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "volleyball_data"
OUT = ROOT / "artifacts"

# Chia theo TRAN DAU (video_id), khong chia theo anh.
# Cung mot tran = cung san, cung anh sang, cung khan gia -> neu nam o ca Train va Test
# thi model "thuoc" nen san thay vi hoc nhan dien nguoi/bong (data leakage).
VAL_VIDEOS = {"majXaA68z", "maNen4zOD"}
TEST_VIDEOS = {"mak0QV2pk"}
CLASSES = ["Player", "Referee", "Ball"]


def parse_name(stem):
    """'mak0QV2pk-2051.355_2' -> ('mak0QV2pk', 2051.355, 2)"""
    video_id, rest = stem.split("-", 1)
    ts, tile = rest.rsplit("_", 1)
    return video_id, float(ts), int(tile)


def read_boxes(txt_path):
    """Doc file nhan YOLO. File rong (0 dong) tra ve [] - day la mau am hop le."""
    boxes = []
    for line in txt_path.read_text().splitlines():
        if line.strip():
            c, xc, yc, w, h = line.split()
            boxes.append((int(c), float(xc), float(yc), float(w), float(h)))
    return boxes


def audit():
    """Quet toan bo du lieu, tra ve thong ke + danh sach stem."""
    stems = sorted(p.stem for p in DATA.glob("*.jpg"))
    per_video = defaultdict(lambda: {"tiles": 0, "boxes": Counter()})
    empty = []
    for s in stems:
        video_id, _, _ = parse_name(s)
        boxes = read_boxes(DATA / f"{s}.txt")
        if not boxes:
            empty.append(s)
        per_video[video_id]["tiles"] += 1
        for b in boxes:
            per_video[video_id]["boxes"][CLASSES[b[0]]] += 1
    return stems, per_video, empty


def assign_group_split(stems):
    """video_id -> train/val/test."""
    split = {"train": [], "val": [], "test": []}
    for s in stems:
        v = parse_name(s)[0]
        key = "val" if v in VAL_VIDEOS else "test" if v in TEST_VIDEOS else "train"
        split[key].append(s)
    return split


def assign_random_split(stems, seed=42):
    """Chia ngau nhien theo anh, CHI de lam thi nghiem chung minh leak (ablation)."""
    rng = random.Random(seed)
    shuffled = stems[:]
    rng.shuffle(shuffled)
    n_val, n_test = round(len(stems) * 0.2), round(len(stems) * 0.1)
    return {
        "val": shuffled[:n_val],
        "test": shuffled[n_val:n_val + n_test],
        "train": shuffled[n_val + n_test:],
    }


def write_split(name, split):
    """Tao cau truc YOLO bang symlink: artifacts/splits/<name>/{images,labels}/<set>/.

    Ultralytics tim nhan bang cach doi '/images/' thanh '/labels/' trong duong dan anh,
    nen bat buoc giu cau truc images/ + labels/. Symlink => khong nhan doi dung luong.
    """
    base = OUT / "splits" / name
    for subset, stems in split.items():
        for kind, ext in (("images", "jpg"), ("labels", "txt")):
            d = base / kind / subset
            d.mkdir(parents=True, exist_ok=True)
            for old in d.iterdir():  # chay lai nhieu lan van sach
                old.unlink()
            for s in stems:
                (d / f"{s}.{ext}").symlink_to(DATA / f"{s}.{ext}")
    yaml = (
        f"path: {base}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    (base / "data.yaml").write_text(yaml)


def videos_of(stems):
    return {parse_name(s)[0] for s in stems}


if __name__ == "__main__":
    stems, per_video, empty = audit()
    total_boxes = Counter()
    for v in per_video.values():
        total_boxes.update(v["boxes"])

    print(f"anh: {len(stems)} | video: {len(per_video)} | box: {sum(total_boxes.values())}")
    print("box theo lop:", dict(total_boxes))
    print("file nhan rong:", empty)
    assert len(stems) == 1051
    assert sum(total_boxes.values()) == 5190
    assert len(empty) == 2

    group = assign_group_split(stems)
    rand = assign_random_split(stems)

    # Gate 1: khong video nao chung giua cac tap (group split)
    tr, va, te = (videos_of(group[k]) for k in ("train", "val", "test"))
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    assert sum(len(v) for v in group.values()) == len(stems)

    # Chung minh random split bi leak: so video xuat hien o ca train va test/val
    leaked = videos_of(rand["train"]) & (videos_of(rand["val"]) | videos_of(rand["test"]))
    print(f"group split: train={len(group['train'])} val={len(group['val'])} test={len(group['test'])} | video chung = 0")
    print(f"random split: {len(leaked)}/{len(per_video)} video bi chung giua train va val/test -> LEAK")
    assert leaked

    write_split("group_split", group)
    write_split("random_split", rand)

    (OUT / "eval").mkdir(parents=True, exist_ok=True)
    (OUT / "eval" / "dataset_audit.json").write_text(json.dumps({
        "images": len(stems),
        "boxes": dict(total_boxes),
        "empty_labels": empty,
        "per_video": {v: {"tiles": d["tiles"], "boxes": dict(d["boxes"])} for v, d in per_video.items()},
        "group_split": {k: len(v) for k, v in group.items()},
        "random_split": {k: len(v) for k, v in rand.items()},
    }, indent=2))
    print("OK -> artifacts/splits/{group_split,random_split}, artifacts/eval/dataset_audit.json")
