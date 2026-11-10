"""Test nhanh cho pipeline: pytest tests/test_pipeline.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data_split import assign_group_split, audit, parse_name, read_boxes, videos_of, DATA  # noqa: E402
from panorama import gt_boxes_global, index_frames, merge_seam_boxes  # noqa: E402
from tactics import get_H, project, voronoi_half  # noqa: E402


def test_group_split_has_no_shared_video():
    stems, _, empty = audit()
    sp = assign_group_split(stems)
    assert videos_of(sp["train"]).isdisjoint(videos_of(sp["val"]) | videos_of(sp["test"]))
    assert videos_of(sp["val"]).isdisjoint(videos_of(sp["test"]))
    assert len(empty) == 2


def test_empty_label_file_parses():
    assert read_boxes(DATA / "maJM7QoN7-2935.532_4.txt") == []


def test_seam_merge_joins_split_box_and_keeps_others():
    left = [0, 700, 100, 768, 300, 0.9]
    right = [0, 768, 110, 820, 310, 0.8]
    far = [0, 1000, 100, 1080, 300, 0.7]
    out = merge_seam_boxes([left, right, far])
    assert len(out) == 2
    m = min(out, key=lambda b: b[1])
    assert (m[1], m[3]) == (700, 820)


def test_seam_merge_on_real_data():
    frames = index_frames()
    removed = sum(len(g) - len(merge_seam_boxes(g)) for g in (gt_boxes_global(v, t, frames) for v, t in frames))
    assert removed >= 50


def test_homography_maps_calibration_corners():
    H = get_H("mak0QV2pk")
    from tactics import _CALIB
    c = _CALIB["mak0QV2pk"]
    out = project(H, c["img_pts"])
    assert np.abs(out - np.array(c["court_pts"])).max() < 1e-3


def test_voronoi_conserves_half_court_area():
    pts = np.array([[1.0, 2.0], [4.0, 7.0], [8.0, 4.0]])
    areas, dead, cov = voronoi_half(pts, 0.0)
    assert abs(sum(areas) - 81.0) < 1e-6 and 0 <= cov <= 100


if __name__ == "__main__":  # chay khong can pytest: python3 tests/test_pipeline.py
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
