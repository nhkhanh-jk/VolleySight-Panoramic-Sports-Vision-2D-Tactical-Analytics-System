"""Phase 3 - Homography san 18x9m, phan doi theo mau ao (CIE-Lab + K-Means), Voronoi coverage.

python3 src/tactics.py   # self-check tren tran test (mak0QV2pk) bang nhan that
"""
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans

from data_split import OUT
from panorama import gt_boxes_global, index_frames, load_backgrounds, merge_seam_boxes, stitch

COURT_W, COURT_H, CELL = 18.0, 9.0, 0.1
DEAD_ZONE_M = 2.5          # o san cach moi nguoi phong thu > 2.5m = vung trong
LIBERO_TAU = 22.0          # khoang cach Lab toi tam doi > TAU => nghi libero (heuristic, chua co GT)
ON_COURT = (-3.0, 21.0, -2.0, 11.0)  # x0, x1, y0, y1: san + vung giao bong, loai khan gia/ban huan luyen

_CALIB = json.loads((OUT / "homography.json").read_text())


def get_H(video_id):
    c = _CALIB[video_id]
    H, _ = cv2.findHomography(np.float32(c["img_pts"]), np.float32(c["court_pts"]))
    return H


def project(H, pts):
    """pts (n,2) pixel panorama -> (n,2) met tren san."""
    if len(pts) == 0:
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), H).reshape(-1, 2)


def torso_lab(img, box):
    """Median Lab cua vung than tren (bo chan/tay/nen). Median chong nhieu da va nen."""
    _, x1, y1, x2, y2, _ = box
    w, h = x2 - x1, y2 - y1
    crop = img[int(y1 + 0.18 * h):int(y1 + 0.50 * h), int(x1 + 0.25 * w):int(x2 - 0.25 * w)]
    if crop.size == 0:
        return np.zeros(3)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return np.median(lab, axis=0)


def voronoi_half(points, x0):
    """Luoi 90x90 (0.1m) cho nua san bat dau tai x0. Tra ve (dien tich moi nguoi m2, dead_zone_m2, coverage%)."""
    xs = x0 + CELL * (np.arange(90) + 0.5)
    ys = CELL * (np.arange(90) + 0.5)
    gx, gy = np.meshgrid(xs, ys)
    cells = np.stack([gx.ravel(), gy.ravel()], 1)
    if len(points) == 0:
        return [], 81.0, 0.0
    d = np.linalg.norm(cells[:, None, :] - points[None, :, :], axis=2)
    owner, mind = d.argmin(1), d.min(1)
    areas = np.bincount(owner, minlength=len(points)) * CELL * CELL
    dead = float((mind > DEAD_ZONE_M).sum() * CELL * CELL)
    return areas.tolist(), dead, 100.0 * (1 - dead / 81.0)


def game_state(courts):
    """Heuristic: 1 nguoi sau duong cuoi san, trong be ngang san => PRE_SERVE; <6 nguoi tren san => DEAD_BALL."""
    behind = [p for p in courts if (p[0] < -0.5 or p[0] > 18.5) and 0 <= p[1] <= 9]
    inside = [p for p in courts if 0 <= p[0] <= 18 and 0 <= p[1] <= 9]
    if len(inside) < 6:
        return "DEAD_BALL"
    return "PRE_SERVE" if len(behind) == 1 else "IN_RALLY"


def analyze(video_id, timestamp, boxes, pano):
    """boxes: [cls,x1,y1,x2,y2,score] da merge seam. Tra ve FrameTacticalState (dict)."""
    H = get_H(video_id)
    x0, x1, y0, y1 = ON_COURT
    players = [b for b in boxes if b[0] == 0 and (b[3] - b[1]) < 160]
    feet = np.array([[(b[1] + b[3]) / 2, b[4]] for b in players]).reshape(-1, 2)
    courts = project(H, feet)
    keep = [i for i, c in enumerate(courts) if x0 <= c[0] <= x1 and y0 <= c[1] <= y1]
    players, courts = [players[i] for i in keep], courts[keep]

    labs = np.array([torso_lab(pano, b) for b in players]).reshape(-1, 3)
    team = ["TEAM_LEFT" if c[0] < 9.0 else "TEAM_RIGHT" for c in courts]
    is_libero = [False] * len(players)
    if len(players) >= 4:
        # L (do sang) nhay voi bong/anh sang => giam trong so; a,b (sac do) phan biet ao doi tot hon
        feats = labs * np.array([0.5, 1.0, 1.0], np.float32)
        left, right = courts[:, 0] < 9.0, courts[:, 0] >= 9.0
        if left.sum() >= 2 and right.sum() >= 2:
            init = np.stack([feats[left].mean(0), feats[right].mean(0)])
            km = KMeans(2, init=init, n_init=1, random_state=0).fit(feats)
        else:
            km = KMeans(2, n_init=10, random_state=0).fit(feats)
        dist = np.linalg.norm(feats - km.cluster_centers_[km.labels_], axis=1)
        for side_idx, side_name in enumerate(("TEAM_LEFT", "TEAM_RIGHT")):
            idx = [i for i, v in enumerate(team) if v == side_name and 0.0 <= courts[i, 0] <= 18.0]
            if len(idx) >= 2:
                # Neu 1 cau thu trong san co mau ao roi vao cum mau doi phuong hoac lech xa tam (> LIBERO_TAU) => Libero
                maj_cluster = int(np.bincount(km.labels_[idx]).argmax())
                outliers = [i for i in idx if km.labels_[i] != maj_cluster or dist[i] > LIBERO_TAU]
                if outliers:
                    j = max(outliers, key=lambda i: np.linalg.norm(feats[i] - km.cluster_centers_[maj_cluster]))
                    is_libero[j] = True

    out_players, cc = [], {}
    for side, xh in (("LEFT", 0.0), ("RIGHT", 9.0)):
        idx = [i for i, t in enumerate(team) if t == f"TEAM_{side}"]
        # Voronoi chi tinh nguoi cua doi do dang dung trong nua san cua ho
        in_half = [i for i in idx if xh - 1 <= courts[i, 0] <= xh + 10]
        areas, dead, cov = voronoi_half(courts[in_half], xh)
        cc[f"team_{side.lower()}_coverage_pct"] = round(cov, 1)
        cc[f"{side.lower()}_dead_zone_m2"] = round(dead, 1)
        area_of = dict(zip(in_half, areas))
        for i in idx:
            out_players.append({
                "id": i, "bbox_pano": [round(v, 1) for v in players[i][1:5]],
                "foot_pano": [round((players[i][1] + players[i][3]) / 2, 1), round(players[i][4], 1)],
                "court_2d_m": [round(float(courts[i, 0]), 2), round(float(courts[i, 1]), 2)],
                "team": f"TEAM_{side}", "is_libero": is_libero[i],
                "voronoi_area_m2": round(area_of.get(i, 0.0), 2),
            })

    balls = [b for b in boxes if b[0] == 2]
    ball = None
    if balls:
        b = max(balls, key=lambda b: b[5])
        ball = project(H, [[(b[1] + b[3]) / 2, (b[2] + b[4]) / 2]])[0].round(2).tolist()  # bong bay: chi tuong doi
    return {"video_id": video_id, "timestamp": timestamp, "game_state": game_state([p["court_2d_m"] for p in out_players]),
            "ball_2d_m": ball, "players": out_players, "court_control": cc}


def radar(state, scale=50, margin=3):
    """Ve san 2D + Voronoi + vung trong + nguoi choi + bong."""
    W, Hh = int((COURT_W + 2 * margin) * scale), int((COURT_H + 2 * margin) * scale)
    img = np.full((Hh, W, 3), (60, 90, 60), np.uint8)
    px = lambda x, y: (int((x + margin) * scale), int((y + margin) * scale))
    cv2.rectangle(img, px(0, 0), px(COURT_W, COURT_H), (90, 130, 200), -1)
    colors = {"TEAM_LEFT": (255, 120, 40), "TEAM_RIGHT": (60, 60, 230)}
    for side, xh in (("LEFT", 0.0), ("RIGHT", 9.0)):
        pts = [p for p in state["players"] if p["team"] == f"TEAM_{side}" and xh - 1 <= p["court_2d_m"][0] <= xh + 10]
        if not pts:
            continue
        xs = xh + CELL * (np.arange(90) + 0.5)
        ys = CELL * (np.arange(90) + 0.5)
        gx, gy = np.meshgrid(xs, ys)
        P = np.array([p["court_2d_m"] for p in pts])
        d = np.linalg.norm(np.stack([gx, gy], -1)[:, :, None, :] - P[None, None], axis=3)
        owner, mind = d.argmin(2), d.min(2)
        shade = np.array(colors[f"TEAM_{side}"], np.float32)
        for r in range(0, 90):
            for c in range(0, 90):
                col = shade * (0.35 + 0.1 * (owner[r, c] % 4)) if mind[r, c] <= DEAD_ZONE_M else (20, 20, 20)
                a, b = px(xh + c * CELL, r * CELL), px(xh + (c + 1) * CELL, (r + 1) * CELL)
                cv2.rectangle(img, a, b, tuple(int(v) for v in col), -1)
    cv2.rectangle(img, px(0, 0), px(COURT_W, COURT_H), (255, 255, 255), 2)
    cv2.line(img, px(9, 0), px(9, COURT_H), (255, 255, 255), 3)
    for x in (6, 12):
        cv2.line(img, px(x, 0), px(x, COURT_H), (255, 255, 255), 1)
    for p in state["players"]:
        c = colors[p["team"]]
        cv2.circle(img, px(*p["court_2d_m"]), 9, c, -1)
        cv2.circle(img, px(*p["court_2d_m"]), 9, (0, 255, 255) if p["is_libero"] else (255, 255, 255), 2)
    if state["ball_2d_m"]:
        cv2.circle(img, px(*state["ball_2d_m"]), 6, (0, 255, 255), -1)
    return img


def export_demo_video(
    video_path=None,
    out_path=None,
    start_frame=135,
    end_frame=405,
    conf=0.25,
    imgsz=1536,
):
    """Xuat video .mp4 ghep Tren (Panorama + Box + Quy dao bong) va Duoi (Radar 2D Voronoi + Bang chi so).
    Chay: python3 src/tactics.py video [start_frame] [end_frame]
    """
    from collections import deque
    from detector import _device, weights
    from ultralytics import YOLO

    root = OUT.parent
    video_path = Path(video_path or (root / "output.mp4"))
    out_path = Path(out_path or (OUT / "eval" / "volleysight_demo.mp4"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights("group_split")))
    dev = _device()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_top = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    H_bot = 580
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H_top + H_bot)
    )

    team_bgr = {"TEAM_LEFT": (255, 140, 40), "TEAM_RIGHT": (60, 60, 235)}
    corners = np.int32(_CALIB["output.mp4"]["img_pts"]).reshape((-1, 1, 2))
    ball_trail = deque(maxlen=14)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for fidx in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        ts = round(fidx / fps, 2)
        r = model.predict(frame, imgsz=imgsz, conf=conf, device=dev, verbose=False)[0]
        boxes = [
            [int(c), x1, y1, x2, y2, s]
            for (x1, y1, x2, y2), c, s in zip(
                r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()
            )
        ]
        state = analyze("output.mp4", ts, boxes, frame)

        # 1. Top panel: Annotated Panorama + court boundary + ball trajectory
        top = frame.copy()
        cv2.polylines(top, [corners], True, (255, 255, 255), 2, cv2.LINE_AA)
        for p in state["players"]:
            x1, y1, x2, y2 = map(int, p["bbox_pano"])
            col = (0, 255, 255) if p["is_libero"] else team_bgr[p["team"]]
            cv2.rectangle(top, (x1, y1), (x2, y2), col, 3)
            tag = f"{'Libero' if p['is_libero'] else p['team'].split('_')[1]} {p['voronoi_area_m2']:.0f}m2"
            cv2.putText(top, tag, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

        balls = [b for b in boxes if b[0] == 2]
        if balls:
            bb = max(balls, key=lambda x: x[5])
            bx, by = int((bb[1] + bb[3]) / 2), int((bb[2] + bb[4]) / 2)
            if ball_trail and np.hypot(bx - ball_trail[-1][0], by - ball_trail[-1][1]) > 260:
                ball_trail.clear()
            ball_trail.append((bx, by))
            cv2.rectangle(top, (int(bb[1]), int(bb[2])), (int(bb[3]), int(bb[4])), (0, 0, 255), 3)
            cv2.putText(top, f"Ball {bb[5]:.2f}", (int(bb[1]), max(22, int(bb[2]) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            if ball_trail:
                ball_trail.popleft()
        for k in range(1, len(ball_trail)):
            cv2.line(top, ball_trail[k - 1], ball_trail[k], (0, 215, 255), 3, cv2.LINE_AA)

        for c, x1, y1, x2, y2, s in boxes:
            if c == 1:
                cv2.rectangle(top, (int(x1), int(y1)), (int(x2), int(y2)), (0, 215, 255), 2)

        # 2. Bottom panel: Left = 2D Voronoi Radar, Right = Tactical Telemetry HUD
        bot = np.full((H_bot, W, 3), (24, 28, 34), np.uint8)
        rad = radar(state, scale=42, margin=2)
        rh, rw = rad.shape[:2]
        ry = (H_bot - rh) // 2
        rx = 40
        bot[ry:ry + rh, rx:rx + rw] = rad

        # HUD Panel on right side of bottom bar
        hx = rx + rw + 60
        cc = state["court_control"]
        cv2.putText(bot, "VOLLEYSIGHT - 2D SPATIAL TACTICAL ANALYTICS", (hx, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bot, f"Time: {ts:5.2f}s (Frame {fidx})   |   Match State: {state['game_state']}", (hx, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 255), 2, cv2.LINE_AA)

        for idx_s, (label, cov_k, dead_k, col) in enumerate([
            ("TEAM LEFT (Red/White)", "team_left_coverage_pct", "left_dead_zone_m2", (255, 140, 40)),
            ("TEAM RIGHT (Yellow/Black)", "team_right_coverage_pct", "right_dead_zone_m2", (60, 60, 235)),
        ]):
            y0 = 185 + idx_s * 125
            cov = cc[cov_k]
            dead = cc[dead_k]
            cv2.putText(bot, f"{label}: Coverage {cov:4.1f}%  |  Dead Zone (>2.5m): {dead:4.1f} m2",
                        (hx, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (230, 230, 230), 2, cv2.LINE_AA)
            bar_w = 560
            cv2.rectangle(bot, (hx, y0 + 18), (hx + bar_w, y0 + 48), (55, 60, 70), -1)
            cv2.rectangle(bot, (hx, y0 + 18), (hx + int(bar_w * min(cov, 100) / 100.0), y0 + 48), col, -1)
            cv2.rectangle(bot, (hx, y0 + 18), (hx + bar_w, y0 + 48), (180, 180, 180), 1)

        cv2.putText(bot, "Legend: [Blue] Team Left  [Red] Team Right  [Yellow] Libero  [Dark] Dead Zone >2.5m",
                    (hx, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (175, 185, 195), 1, cv2.LINE_AA)

        writer.write(np.vstack([top, bot]))
        if (fidx - start_frame) % 45 == 0:
            print(f"Rendered frame {fidx}/{end_frame} ({ts:.1f}s) - {state['game_state']}")

    cap.release()
    writer.release()
    print(f"OK -> {out_path}")
    return out_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "video":
        sf = int(sys.argv[2]) if len(sys.argv) > 2 else 135
        ef = int(sys.argv[3]) if len(sys.argv) > 3 else 405
        export_demo_video(start_frame=sf, end_frame=ef)
        sys.exit(0)
    frames = index_frames()
    bg = load_backgrounds()
    video = "mak0QV2pk"
    ok = tot = 0
    best, best_n = None, 0
    cov_l, cov_r = [], []
    for (v, ts) in sorted(frames):
        if v != video:
            continue
        boxes = merge_seam_boxes(gt_boxes_global(v, ts, frames))
        if sum(1 for b in boxes if b[0] == 0) < 8:
            continue
        pano, _ = stitch(v, ts, frames, bg)
        s = analyze(v, ts, boxes, pano)
        n_l = sum(p["team"] == "TEAM_LEFT" for p in s["players"])
        n_r = len(s["players"]) - n_l
        tot += 1
        ok += (4 <= n_l <= 7 and 4 <= n_r <= 7)   # moi doi ~6 nguoi tren san
        cov_l.append(s["court_control"]["team_left_coverage_pct"])
        cov_r.append(s["court_control"]["team_right_coverage_pct"])
        if len(s["players"]) > best_n:
            best, best_n = (s, pano), len(s["players"])
        # Voronoi bao toan dien tich: tong dien tich moi doi == 81 m2
        for side in ("TEAM_LEFT", "TEAM_RIGHT"):
            a = sum(p["voronoi_area_m2"] for p in s["players"] if p["team"] == side)
            assert a == 0 or abs(a - 81.0) < 0.5, (side, a)
    print(f"{video}: {tot} frame >=8 nguoi | chia doi hop ly (4-7 moi ben): {ok}/{tot}")
    print(f"coverage TB trai={np.mean(cov_l):.1f}% phai={np.mean(cov_r):.1f}% (dead zone > {DEAD_ZONE_M} m)")
    assert tot > 0 and ok / tot >= 0.6
    s, pano = best
    (OUT / "eval").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "eval" / "tactics_radar.jpg"), radar(s))
    print(json.dumps(s["court_control"]), s["game_state"])
    print("OK -> artifacts/eval/tactics_radar.jpg")
