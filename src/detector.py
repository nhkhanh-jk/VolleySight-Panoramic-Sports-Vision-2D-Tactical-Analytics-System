"""Phase 2 - Train YOLO, danh gia, va 2 thi nghiem ablation.

python3 src/detector.py train   # train group_split (model chinh) + random_split (chi de chung minh leak)
python3 src/detector.py eval    # ablation -> artifacts/eval/ablation_results.md
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from data_split import CLASSES, DATA, OUT, parse_name, read_boxes
from panorama import SEAMS, TILE, merge_seam_boxes

RUNS = OUT / "runs"
SPLITS = OUT / "splits"
IMGSZ = 768


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# A100/CUDA: batch 16, 40 epoch. Mac MPS: batch 8, 25 epoch (nhe hon cho may ca nhan)
_GPU = _device() == "cuda"
EPOCHS, BATCH = (40, 16) if _GPU else (25, 8)


def weights(name):
    return RUNS / name / "weights" / "best.pt"


def train(name):
    from ultralytics import YOLO
    model = YOLO("yolo11s.pt")
    # copy_paste/mosaic/scale: tang du lieu giup bong nho (~33px) de hoc hon
    model.train(
        data=str(SPLITS / name / "data.yaml"), imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH,
        device=_device(), project=str(RUNS), name=name, exist_ok=True, workers=2,
        mosaic=1.0, copy_paste=0.3, scale=0.5, patience=15, seed=0,
    )


def val(name, split_name, split="test", **kw):
    """Tra ve dict metric (per-class AP50, recall, mAP50, mAP50-95)."""
    from ultralytics import YOLO
    m = YOLO(str(weights(name))).val(
        data=str(SPLITS / split_name / "data.yaml"), split=split, device=_device(),
        batch=16, verbose=False, plots=False, **kw)
    b = m.box
    return {
        "mAP50": b.map50, "mAP50-95": b.map,
        "AP50": dict(zip(CLASSES, b.ap50)),
        "recall": dict(zip(CLASSES, b.r)),
    }


def iou(a, b):
    ix = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    iy = max(0, min(a[4], b[4]) - max(a[2], b[2]))
    i = ix * iy
    u = (a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - i
    return i / u if u else 0


def match(preds, gts, thr=0.5):
    """Greedy theo score. Tra ve (tp, fp, fn). Chi so khop khi cung lop va IoU>=thr."""
    used, tp = set(), 0
    for p in sorted(preds, key=lambda b: -b[5]):
        best, bj = 0, None
        for j, g in enumerate(gts):
            if j not in used and g[0] == p[0] and (v := iou(p, g)) > best:
                best, bj = v, j
        if bj is not None and best >= thr:
            used.add(bj)
            tp += 1
    return tp, len(preds) - tp, len(gts) - tp


def predict_frame(model, tiles, conf=0.25):
    """tiles: {tile_idx: duong dan anh}. Tra ve box toan cuc [cls,x1,y1,x2,y2,score] (chua merge seam)."""
    pred = []
    for t, path in tiles.items():
        r = model.predict(str(path), imgsz=IMGSZ, conf=conf, device=_device(), verbose=False)[0]
        for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            pred.append([int(c), x1 + t * TILE, y1, x2 + t * TILE, y2, s])
    return pred


def seam_ablation(name="group_split"):
    """Test tiles -> box toan cuc. So sanh pred tho vs pred da gop seam, doi chieu GT da gop
    (GT: moi vat the la 1 box, nen nua-nguoi o 2 tile cua pred tho bi tinh la FP/FN)."""
    from ultralytics import YOLO
    model = YOLO(str(weights(name)))
    test_dir = SPLITS / "group_split" / "images" / "test"
    frames = defaultdict(dict)
    for p in sorted(test_dir.glob("*.jpg")):
        v, ts, t = parse_name(p.stem)
        frames[(v, ts)][t] = p
    totals = {"raw": np.zeros(3), "merged": np.zeros(3)}
    for key, tiles in frames.items():
        gt = []
        for t, path in tiles.items():
            for c, xc, yc, w, h in read_boxes(DATA / f"{path.stem}.txt"):
                gt.append([c, (xc - w / 2) * TILE + t * TILE, (yc - h / 2) * TILE,
                           (xc + w / 2) * TILE + t * TILE, (yc + h / 2) * TILE, 1.0])
        pred = predict_frame(model, tiles)
        gt_m = merge_seam_boxes(gt)
        totals["raw"] += match(pred, gt_m)
        totals["merged"] += match(merge_seam_boxes(pred), gt_m)
    res = {}
    for k, (tp, fp, fn) in totals.items():
        res[k] = {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "fp": int(fp)}
    return res


def evaluate():
    rows = []
    g_test = val("group_split", "group_split")
    r_test = val("random_split", "random_split")
    r_on_g = val("random_split", "group_split")  # model train random, thu tren tran chua tung thay
    rows.append("## Ablation 1 - Scene leakage (test set)\n")
    rows.append("| Train split | Test set | mAP@50 | mAP@50-95 |\n|---|---|---|---|")
    rows.append(f"| random | random test (leak) | {r_test['mAP50']:.3f} | {r_test['mAP50-95']:.3f} |")
    rows.append(f"| random | group test (tran moi) | {r_on_g['mAP50']:.3f} | {r_on_g['mAP50-95']:.3f} |")
    rows.append(f"| group | group test (tran moi) | {g_test['mAP50']:.3f} | {g_test['mAP50-95']:.3f} |\n")

    rows.append("## Per-class (group_split, test)\n")
    rows.append("| Class | AP@50 | Recall |\n|---|---|---|")
    for c in CLASSES:
        rows.append(f"| {c} | {g_test['AP50'][c]:.3f} | {g_test['recall'][c]:.3f} |")

    rows.append("\n## Ablation 2a - Small ball inference (group_split, test)\n")
    rows.append("| Inference | Ball AP@50 | Ball Recall | mAP@50 |\n|---|---|---|---|")
    for label, kw in [("standard imgsz=768", {}), ("upscale imgsz=1152", {"imgsz": 1152}),
                      ("TTA augment=True", {"augment": True})]:
        m = g_test if not kw else val("group_split", "group_split", **kw)
        rows.append(f"| {label} | {m['AP50']['Ball']:.3f} | {m['recall']['Ball']:.3f} | {m['mAP50']:.3f} |")

    s = seam_ablation()
    rows.append("\n## Ablation 2b - Seam merging (panorama, IoU 0.5, conf 0.25)\n")
    rows.append("| Predictions | Precision | Recall | FP |\n|---|---|---|---|")
    for k in ("raw", "merged"):
        rows.append(f"| {k} | {s[k]['precision']:.3f} | {s[k]['recall']:.3f} | {s[k]['fp']} |")

    (OUT / "eval").mkdir(parents=True, exist_ok=True)
    (OUT / "eval" / "ablation_results.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "train":
        for n in ("group_split", "random_split"):
            train(n)
    else:
        evaluate()
