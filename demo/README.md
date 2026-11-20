# Interactive spoke-tracker explorer

This is an algorithm-level walkthrough, not a presentation or a live tracking service. Select one of 39 adjacent observation pairs and walk through:

1. The previous observed mask and its connected spoke regions.
2. The orbital-motion prediction for the actual time gap.
3. The new observed mask.
4. Every prediction–observation candidate pair, with its actual distance, shape score, and cost.
5. Greedy matching, one sorted pair at a time—or with automatic stepping.
6. The observed mask after track IDs and colors propagate.

The browser reads compact precomputed data and does not reimplement the motion model, matching cost, or tracker. All 39 transitions come from the real Cassini masks in `data/masks/` and their coordinate metadata.

## Open the demo

From the repository root:

```bash
python3 -m http.server 8000
```

Open [http://localhost:8000/demo/](http://localhost:8000/demo/). An HTTP server is required because the page lazily loads JSON for each selected transition.

## Regenerate the intermediate data

The checked-in JSON in `demo/data/` is ready to use. Regenerate it after changing tracker behavior or input masks:

```bash
venv/bin/python -B demo/build_demo_data.py
```

The generator runs the production `Tracker`, delegates match decisions to `Tracker._match_spokes()`, and records pair metrics and object geometry along the way. It uses the same grid-snapping function as `track_folder.py` for visualization. Matching itself occurs on continuous physical coordinates, not these snapped pixels.

The first transition is roughly 0.5 MB; other transitions load only when selected.

## Files

- `index.html`: six-stage explorer and accessible controls
- `styles.css`: responsive workbench styling
- `app.js`: stage navigation, pair inspection, canvas rendering, and greedy-step controls
- `build_demo_data.py`: production-tracker instrumentation and JSON generation
- `data/index.json`: transition summaries
- `data/transition_XX.json`: object points, candidate costs, and accepted matches for one adjacent frame pair
