"""Phase 4 - Dashboard Streamlit: Panorama Dataset + Video Thực Tế (output.mp4) + Radar 2D/Voronoi.

streamlit run app.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from data_split import DATA, OUT  # noqa: E402
from detector import _device, predict_frame, weights  # noqa: E402
from panorama import gt_boxes_global, index_frames, load_backgrounds, merge_seam_boxes, stitch  # noqa: E402
from tactics import _CALIB, analyze, radar  # noqa: E402

TEAM_BGR = {"TEAM_LEFT": (255, 120, 40), "TEAM_RIGHT": (60, 60, 230)}
VIDEO_PATH = ROOT / "output.mp4"

st.set_page_config(page_title="VolleySight", layout="wide")


@st.cache_data
def load_index():
    return index_frames()


@st.cache_data
def load_bg():
    return load_backgrounds()


@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO(str(weights("group_split")))


def render_annotated_frame(pano, boxes, state, real=None, draw_seams=True):
    vis = pano.copy()
    if real is not None:
        for t in range(5):
            if not real[t]:
                cv2.rectangle(vis, (t * 768, 0), (t * 768 + 767, 767), (0, 0, 0), 6)
    for p in state["players"]:
        x1, y1, x2, y2 = map(int, p["bbox_pano"])
        col = (0, 255, 255) if p["is_libero"] else TEAM_BGR[p["team"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3)
        lbl = f"{'Libero' if p['is_libero'] else p['team'].split('_')[1]} ({p['voronoi_area_m2']:.0f}m2)"
        cv2.putText(vis, lbl, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    for c, x1, y1, x2, y2, s in boxes:
        if c == 1:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 3)
        elif c == 2:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            cv2.putText(vis, f"Ball {s:.2f}", (int(x1), max(20, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    if draw_seams:
        for x in (768, 1536, 2304, 3072):
            cv2.line(vis, (x, 0), (x, vis.shape[0] - 1), (255, 255, 0), 1)
    return vis


st.title("VolleySight - Panoramic Sports Vision & 2D Spatial Tactical Analytics")
opts = ["Video Thuc Te (output.mp4)", "Dataset Panorama (volleyball_data)"] if VIDEO_PATH.exists() else ["Dataset Panorama (volleyball_data)"]
mode = st.sidebar.radio("Che do du lieu", opts)

if mode.startswith("Video"):
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = st.sidebar.slider("Frame", 0, max(0, n_frames - 1), 0, step=15)
    conf_thr = st.sidebar.slider("Confidence threshold", 0.15, 0.70, 0.25, step=0.05)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, pano = cap.read()
    cap.release()
    ts = round(frame_idx / fps, 2)
    r = load_model().predict(pano, imgsz=1536, conf=conf_thr, device=_device(), verbose=False)[0]
    boxes = [[int(c), x1, y1, x2, y2, s] for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
    state = analyze("output.mp4", ts, boxes, pano)
    vis = render_annotated_frame(pano, boxes, state, real=None, draw_seams=False)
    video_label = "output.mp4"
else:
    frames, bg = load_index(), load_bg()
    videos = sorted(v for v in {k[0] for k in frames if k[0] in _CALIB})
    video_label = st.sidebar.selectbox("Tran dau (video_id)", videos)
    stamps = sorted(ts for (v, ts) in frames if v == video_label)
    ts = st.sidebar.select_slider("Timestamp (s)", stamps)
    has_model = weights("group_split").exists()
    source = st.sidebar.radio("Nguon box", ["Nhan that (GT)"] + (["Model YOLO"] if has_model else []))
    show_seam = st.sidebar.checkbox("Merge seam", True)
    pano, real = stitch(video_label, ts, frames, bg)
    if source.startswith("Nhan"):
        boxes = gt_boxes_global(video_label, ts, frames)
    else:
        tiles = {t: DATA / f"{s}.jpg" for t, s in frames[(video_label, ts)].items()}
        boxes = predict_frame(load_model(), tiles)
    if show_seam:
        boxes = merge_seam_boxes(boxes)
    state = analyze(video_label, ts, boxes, pano)
    vis = render_annotated_frame(pano, boxes, state, real=real, draw_seams=True)

st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"Frame Panorama ({video_label} @ {ts}s)", use_container_width=True)
c1, c2 = st.columns([3, 2])
c1.image(cv2.cvtColor(radar(state), cv2.COLOR_BGR2RGB), caption="Radar 2D 18x9m + Voronoi (toi = vung trong > 2.5m)", use_container_width=True)
cc = state["court_control"]
c2.metric("Trang thai", state["game_state"])
c2.metric("Coverage doi trai / phai (%)", f"{cc['team_left_coverage_pct']} / {cc['team_right_coverage_pct']}")
c2.metric("Dead zone trai / phai (m2)", f"{cc['left_dead_zone_m2']} / {cc['right_dead_zone_m2']}")
c2.json(state, expanded=False)
st.download_button("Tai JSON", json.dumps(state, indent=1), file_name=f"{video_label}_{ts}.json")
