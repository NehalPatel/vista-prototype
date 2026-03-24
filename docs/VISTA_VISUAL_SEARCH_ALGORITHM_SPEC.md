# VISTA visual search: implemented store vs future retrieval

This document describes (1) what the **vista-prototype** repository **actually implements** today, and (2) a **target** retrieval algorithm for a **future** search service. It replaces earlier text that referenced non-existent modules such as `search_engine.service` or `VisualSearchService`.

---

## Part A — Implemented today (repository)

### Pipeline outputs

- One `detection_results.json` per processed video under `vista-prototype/results/<video_id>/` (see `pipeline.paths`, `get_video_results_paths`).
- Top-level JSON includes `video_id`, `confidence_threshold`, `object_model`, `face_model`, `frames` (array), optional `run_stats`.
- Each element of `frames` includes:
  - `frame` — filename (e.g. `frame_0001.jpg`)
  - `detections` — YOLO outputs: class, color, label, bbox, conf
  - `faces` — list of `{ bbox, confidence, label, recognition_confidence (optional) }` from `pipeline.faces.run_face_detection`
  - `monument` — frame-level classifier output `{ label, confidence, bbox (optional) }` from `pipeline.monuments.run_monument_recognition`

### Optional MongoDB indexing

- **Module:** `pipeline.mongodb_store`
- **Environment:** `MONGODB_URI` or `MONGO_URI`; optional `VISTA_DB_NAME` (default `vista_search`).
- **Collections:** `videos` (one document per video), `frames` (one document per frame with nested `objects`, `faces`, `monument`).
- **Indexes:** include `faces.label`, `objects.class`, `objects.color`, `objects.label`, `monument.label`, plus uniqueness on `video_id` and `(video_id, frame_filename)`.
- **Trigger:** after a successful web run, `index_detection_results_to_mongodb` in `web/app.py` (not the CLI `implementation.py`).

### Not implemented in this repository

- No `search_engine` package, no `VisualSearchService`, no `frame_events` collection (the store uses `frames`).
- No `POST /api/index-video` or `GET /api/search` routes in `web/app.py`.
- No automated IR evaluation harness or labeled query sets.

---

## Part B — Target retrieval design (future work)

The following is a **design specification** for a separate search API or service that reads the same MongoDB (or exported JSON).

### Proposed algorithm

1. Ingest per-video `detection_results.json` (or read from `frames` / `videos`).
2. Normalize labels (trim, lowercase for matching, handle `Maybe:` face prefixes).
3. Parse natural-language queries into slots: person, monument, object class/color, optional time window, optional confidence floor.
4. Run **strict** retrieval (all active slots must match one frame); if zero hits, optionally run **soft** token fallback on labels.
5. Score and rank frames; group by `video_id`.

### Illustrative MongoDB filters (on `frames`)

- Person: `faces` array element match on `label`.
- Monument: `monument.label`.
- Object: `objects` array with matching `class` and/or `color` / `label`.
- Time: `time_sec` within window.

### API sketch (not implemented)

- `GET /api/search?q=...` — would return ranked frame evidence and `source_url` from `videos`.
- `GET /api/search/suggestions` — would list distinct indexed people and monuments.

### Evaluation (future)

- Precision@K, Recall@K, MRR, Success@K require a frozen corpus and relevance judgments.
- Until then, report **processing** metrics from `run_stats` and qualitative inspection.

### Risks and controls

- Face uncertainty (`Maybe:`): preserve in JSON and penalize in future ranking.
- Monument model is frame-level, not object-level: treat as frame evidence.
- Timestamp mapping follows extraction FPS and frame index conventions used when writing `time_sec`.

---

## Thesis alignment

- **Chapter 4** of the thesis describes the implemented architecture and MongoDB shape.
- **Chapter 6** describes Part B as **conceptual** retrieval design, not shipped code.
