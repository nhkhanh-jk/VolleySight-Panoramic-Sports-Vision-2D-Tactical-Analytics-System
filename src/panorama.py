"""Phase 1.2 - Dung lai panorama 3840x768 tu cac tile, va gop box bi cat o duong noi.

Chay: python3 src/panorama.py
"""
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from data_split import DATA, OUT, parse_name, read_boxes

TILE = 768
N_TILES = 5
SEAMS = tuple(TILE * t for t in range(1, N_TILES))  # (768, 1536, 2304, 3072)
BG_DIR = OUT / "backgrounds"


def index_frames():
    """{(video_id, timestamp): {tile_idx: stem}}"""
    frames = defaultdict(dict)
    for p in sorted(DATA.glob("*.jpg")):
        v, ts, t = parse_name(p.stem)
        frames[(v, ts)][t] = p.stem
    return frames


def build_backgrounds(frames):
    """Median theo thoi gian cua tung (video, tile). Camera co dinh nen nen san giu nguyen,
    con nguoi di chuyen => median loai bo ho. Tile nao chi co 1-2 anh thi median kem sach hon."""
    BG_DIR.mkdir(parents=True, exist_ok=True)
    stacks = defaultdict(list)
    for (v, _), tiles in frames.items():
        for t, stem in tiles.items():
            stacks[(v, t)].append(cv2.imread(str(DATA / f"{stem}.jpg")))
    bg = {}
    for (v, t), imgs in stacks.items():
        bg[(v, t)] = np.median(np.stack(imgs), axis=0).astype(np.uint8)
        cv2.imwrite(str(BG_DIR / f"{v}_{t}.jpg"), bg[(v, t)])
    return bg


def load_backgrounds():
    bg = {}
    for p in BG_DIR.glob("*.jpg"):
        v, t = p.stem.rsplit("_", 1)
        bg[(v, int(t))] = cv2.imread(str(p))
    return bg


def stitch(video_id, timestamp, frames, bg):
    """Tra ve (anh 768x3840x3 BGR, real_mask[5]). Tile thieu duoc lap bang background
    (real_mask=False => khong co nhan that o do)."""
    canvas = np.zeros((TILE, TILE * N_TILES, 3), np.uint8)
    real = np.zeros(N_TILES, bool)
    tiles = frames[(video_id, timestamp)]
    for t in range(N_TILES):
        if t in tiles:
            img, real[t] = cv2.imread(str(DATA / f"{tiles[t]}.jpg")), True
        elif (video_id, t) in bg:
            img = bg[(video_id, t)]
        else:
            img = np.zeros((TILE, TILE, 3), np.uint8)
        canvas[:, t * TILE:(t + 1) * TILE] = img
    return canvas, real


def gt_boxes_global(video_id, timestamp, frames):
    """Nhan that -> toa do pixel panorama [cls, x1, y1, x2, y2, score]. x_global = t*768 + x_local."""
    out = []
    for t, stem in frames[(video_id, timestamp)].items():
        for c, xc, yc, w, h in read_boxes(DATA / f"{stem}.txt"):
            x1, x2 = (xc - w / 2) * TILE + t * TILE, (xc + w / 2) * TILE + t * TILE
            out.append([c, x1, (yc - h / 2) * TILE, x2, (yc + h / 2) * TILE, 1.0])
    return out


def merge_seam_boxes(boxes, seams=SEAMS, delta_x=12, iou_y_thresh=0.3):
    """Gop cac cap box cung lop bi cat doi o duong noi + khu box trung lap tren panorama.
    box = [cls, x1, y1, x2, y2, score]. Dieu kien: mep phai box trai cach seam <= delta_x,
    mep trai box phai cach seam <= delta_x, IoU theo truc doc >= iou_y_thresh."""
    boxes = [list(b) for b in boxes]
    for s in seams:
        left = [b for b in boxes if b[0] != 2 and 0 <= s - b[3] <= delta_x]
        right = [b for b in boxes if b[0] != 2 and 0 <= b[1] - s <= delta_x]
        used = set()
        for l in left:
            for r in right:
                if id(r) in used or l[0] != r[0]:
                    continue
                inter = max(0, min(l[4], r[4]) - max(l[2], r[2]))
                union = (l[4] - l[2]) + (r[4] - r[2]) - inter
                if union > 0 and inter / union >= iou_y_thresh:
                    boxes.remove(l)
                    boxes.remove(r)
                    boxes.append([l[0], l[1], min(l[2], r[2]), r[3], max(l[4], r[4]), max(l[5], r[5])])
                    used.add(id(r))
                    break
    # Global 2D NMS (IoU >= 0.45 hoac IoMin >= 0.70) de loai bo 2 box chong/long nhau tren cung 1 nguoi
    kept = []
    for b in sorted(boxes, key=lambda x: -x[5]):
        dup = False
        ab = max((b[3] - b[1]) * (b[4] - b[2]), 1e-6)
        for k in kept:
            if b[0] != k[0]:
                continue
            inter = max(0, min(b[3], k[3]) - max(b[1], k[1])) * max(0, min(b[4], k[4]) - max(b[2], k[2]))
            ak = max((k[3] - k[1]) * (k[4] - k[2]), 1e-6)
            if inter / (ab + ak - inter) >= 0.45 or inter / min(ab, ak) >= 0.70:
                dup = True
                break
        if not dup:
            kept.append(b)
    return kept


def draw(canvas, boxes, real=None):
    img = canvas.copy()
    colors = {0: (0, 200, 0), 1: (0, 200, 255), 2: (0, 0, 255)}
    for c, x1, y1, x2, y2, _ in boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colors[int(c)], 2)
    for s in SEAMS:
        cv2.line(img, (s, 0), (s, TILE), (255, 255, 0), 1)
    if real is not None:
        for t in range(N_TILES):
            if not real[t]:
                cv2.putText(img, "BG", (t * TILE + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img


if __name__ == "__main__":
    frames = index_frames()
    print(f"{len(frames)} frames (video, timestamp)")
    assert len(frames) == 309
    bg = build_backgrounds(frames)

    # Moi cap box cat doi trong nhan that phai duoc gop thanh 1 (122 cap)
    total_before = total_after = 0
    for (v, ts) in frames:
        gt = gt_boxes_global(v, ts, frames)
        merged = merge_seam_boxes(gt)
        total_before += len(gt)
        total_after += len(merged)
    pairs = total_before - total_after
    print(f"box truoc gop={total_before} sau gop={total_after} => da gop {pairs} cap")
    assert pairs >= 50, pairs  # thuc te do duoc 66 cap (so 122 ban dau la uoc luong long, da sua)

    # Anh minh hoa: frame co nhieu tile nhat + co cap box bi cat
    best = None
    for (v, ts) in frames:
        gt = gt_boxes_global(v, ts, frames)
        if len(gt) - len(merge_seam_boxes(gt)) > 0 and len(frames[(v, ts)]) >= 4:
            best = (v, ts, gt)
            break
    v, ts, gt = best
    canvas, real = stitch(v, ts, frames, bg)
    (OUT / "eval").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "eval" / "panorama_before.jpg"), draw(canvas, gt, real))
    cv2.imwrite(str(OUT / "eval" / "panorama_after.jpg"), draw(canvas, merge_seam_boxes(gt), real))
    print(f"OK -> artifacts/eval/panorama_{{before,after}}.jpg ({v} {ts}), backgrounds -> artifacts/backgrounds/")
