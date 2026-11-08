# projectmem - Volleyball

_Last updated: 2026-11-05_

## Project purpose
Replace this placeholder with a concise description of what this project does, who it serves, and the main technologies or runtime assumptions.

## Recent issues
- No issues logged yet.

## Decisions
- Build VolleySight: End-to-End Panoramic Reconstruction, Small-Ball YOLO Detection with Cross-Tile Seam Merging, Unsupervised CIE-Lab Team Clustering, and 2D Homography + Voronoi Court Analytics [PROJECT_MAP.md]
- Mentoring mode: user is learning; explain concept first, user writes code, agent reviews; guide in LEARNING_GUIDE.md [LEARNING_GUIDE.md]

## Notes
- gotcha: volleyball_data contains 768x768 horizontal tiles (_0.._4) from 11 static cameras at 5s intervals; must split Train/Val/Test by video_id prefix to prevent background data leakage [volleyball_data]
- gotcha: unannotated outer tiles (_0, _4) are missing on most timestamps and 2 .txt files are empty; reconstruct full panorama canvas using per-video temporal median background tiles [volleyball_data]
- gotcha: tiles _0.._4 are exact 0-overlap side-by-side crops (x_global = tile_idx * 768 + x_local, full canvas 3840x768) with 122 objects split across adjacent tile seams requiring Cross-Tile Box Merging [volleyball_data]

## Key files
- `LEARNING_GUIDE.md`

## Open questions
- None logged yet.
