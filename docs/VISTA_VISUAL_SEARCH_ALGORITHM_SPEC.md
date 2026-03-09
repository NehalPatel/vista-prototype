# VISTA Visual Search Algorithm Spec

## Objective
- Build a thesis-ready visual search baseline for user-provided videos.
- Target query style: natural language retrieval focused on person and monument entities.
- Core task: return ranked video moments (video id + frame + timestamp) with evidence from face recognition and monument recognition.
- Primary metric: search relevance (Precision@K, Recall@K, MRR, Success@K).

## Current System Audit
- Existing pipeline outputs one `detection_results.json` per video under `vista-prototype/results/<video_id>/` (see `pipeline.paths.RESULTS_DIR`, `get_video_results_paths`).
- Top-level JSON: `video_id`, `confidence_threshold`, `object_model`, `face_model`, `frames` (array), optional `run_stats`.
- Each frame entry in `frames` includes:
  - `frame` — filename (e.g. `frame_0001.jpg`)
  - `detections` — object detections (YOLO; class, color, label, bbox, conf)
  - `faces` — list of `{ "bbox", "confidence", "label", "recognition_confidence" (optional) }` from `pipeline.faces.run_face_detection`
  - `monument` — frame-level classifier output `{ "label", "confidence", "bbox" (optional) }` from `pipeline.monuments.run_monument_recognition`
- Metadata: `metadata.txt` in the same folder contains `source: <url>`; the indexer uses this when `source_url` is not provided.
- Gaps addressed by this spec:
  - Canonical frame-event schema and MongoDB indexing implemented in `search_engine.service`.
  - Natural-language query parsing (person, monument, time, confidence) and strict/soft retrieval implemented.

## Proposed Algorithm
1. Ingest per-video detection JSON.
2. Convert each frame entry to a canonical frame event.
3. Normalize labels and confidences.
4. Upsert video metadata and replace frame events (idempotent re-index).
5. Parse natural-language query into slots.
6. Run strict retrieval first; fallback to soft matching when strict hits are empty.
7. Score and rank frames, then group by video.

## Data Model
### Canonical Frame Event Schema
```json
{
  "video_id": "2vjEKevuV4k",
  "frame": "frame_0007.jpg",
  "timestamp_sec": 6.0,
  "faces": [
    {
      "label": "nehal",
      "display_label": "Nehal",
      "raw_label": "Maybe:Nehal",
      "is_maybe": true,
      "detection_confidence": 0.93,
      "match_confidence": 0.81,
      "bbox": [120, 45, 210, 170]
    }
  ],
  "monument": {
    "label": "taj mahal",
    "display_label": "Taj Mahal",
    "raw_label": "Taj Mahal",
    "confidence": 0.88,
    "bbox": [8, 8, 1260, 710]
  },
  "scores": {
    "face_match_max": 0.81,
    "face_detection_max": 0.93,
    "monument_confidence": 0.88
  },
  "source_url": "https://www.youtube.com/watch?v=2vjEKevuV4k",
  "model_versions": {
    "object_model": "yolov8n",
    "face_model": "buffalo_l",
    "monument_model": "monument_classifier"
  },
  "indexed_at": "UTC datetime"
}
```

### Normalization Rules
- Lowercase + trim labels for canonical matching (`label`).
- Preserve human display form in `display_label`.
- Strip `Maybe:` prefix into canonical label and keep uncertainty in `is_maybe`.
- Map empty/invalid labels to `unknown`.
- Face: use `confidence` as detection confidence, `recognition_confidence` or `match_confidence` as match confidence; both are kept for scoring.

## MongoDB Baseline
- **Connection**: `MONGO_URI` (default `mongodb://localhost:27017`), `MONGO_DB` (default `vista_search`). Implemented in `VisualSearchService` (`search_engine.service`).

### Collections
- `videos`
  - `video_id` (unique)
  - `source_url`
  - `results_path`
  - `model_versions`
  - `indexed_at`
- `frame_events`
  - one document per frame event using the canonical schema above.

### Indexes
- `videos.video_id` (unique)
- `frame_events.video_id + frame` (unique)
- `frame_events.faces.label` (multikey)
- `frame_events.monument.label`
- `frame_events.timestamp_sec`
- `frame_events.video_id`

### Idempotent Re-indexing
- For each index request:
  - Upsert one `videos` record.
  - Delete existing `frame_events` for that `video_id`.
  - Insert rebuilt frame events.
- Result: rescans replace stale frame data without duplicates.

## Query Understanding
### Slot Design
- `person` (optional)
- `monument` (optional)
- `time/window` (optional)
- `min_confidence` (optional)

### Hybrid Parser
- Rule-based extraction first:
  - Match known person labels from indexed data (longest match first).
  - Match known monument labels from indexed data (longest match first).
  - Parse `before`, `after`, `between`/`from ... to` time constraints (seconds).
  - Parse confidence constraints (`conf >= 0.7`, `confidence 0.8`).
- Fallback for unseen entities: if query contains ` at ` and person or monument slot is still empty, split on ` at `; use canonical label of left part as person and right part as monument (e.g. "alex at qutub minar").
- Soft stage: when strict retrieval returns zero hits, tokenize query (tokens ≥3 chars) and regex-match on `faces.label` and `monument.label`.

## Retrieval and Ranking
### Strict Stage
- Build filter from extracted slots:
  - person → `faces` with `$elemMatch: { label: person }`
  - monument → `monument.label == monument`
  - time window → `timestamp_sec` in [min, max]
  - confidence floor → frame matches if **either** `scores.face_match_max >= min_confidence` **or** `scores.monument_confidence >= min_confidence`

### Soft Stage (Fallback)
- Triggered when strict retrieval has zero hits.
- Use token-level regex matching across face and monument labels.

### Ranking
- Score each frame by:
  - exact entity matches (person/monument)
  - partial entity matches
  - confidence signals (`face_match_max`, `monument_confidence`)
- Stable sort by:
  1. score (desc)
  2. timestamp (asc)
  3. frame id (asc)

## API Contracts
### `POST /api/index-video`
- Purpose: index an already processed video into MongoDB (idempotent re-index per video).
- Request JSON:
  - `video_id` or `url` (required): if only `url` is provided, video_id is derived via `extract_video_id_from_url` / `sanitize_id`.
  - `url`: used as `source_url`; if omitted, indexer reads `source: ...` from `results/<video_id>/metadata.txt`.
  - `detection_json_path`: optional; if omitted, path is `results/<video_id>/detection_results.json` (via `get_video_results_paths`).
```json
{
  "video_id": "2vjEKevuV4k",
  "url": "https://www.youtube.com/watch?v=2vjEKevuV4k",
  "detection_json_path": "optional absolute or repo-relative path"
}
```
- Response:
```json
{
  "status": "indexed",
  "video_id": "2vjEKevuV4k",
  "indexed_frames": 180,
  "source_url": "...",
  "detection_json_path": "..."
}
```
- Indexing can also be triggered from `POST /api/process` by setting `index_to_search: true` in the request body; the same indexer runs after processing (or when returning cached results).

### `GET /api/search?q=...`
- Purpose: natural-language visual retrieval.
- Query params:
  - `q` (required)
  - `page` (optional, default `1`)
  - `page_size` (optional, default `20`, max `200`)
- Response:
  - `query`, `mode` (`strict` or `soft`), `slots` (person, monument, min/max_timestamp_sec, min_confidence)
  - `pagination`: `page`, `page_size`, `total_hits`
  - `results`: list of `{ video_id, source_url, frames: [ { frame, timestamp_sec, rank_score, faces, monument, scores } ] }` grouped by video

### `GET /api/search/suggestions`
- Purpose: return known indexed people and monuments for UI autocomplete.
- Query params:
  - `limit` (optional, default `50`, max 500)
- Response:
```json
{
  "people": ["nehal", "virat kohli"],
  "monuments": ["taj mahal", "india gate"],
  "counts": { "people": 2, "monuments": 2 }
}
```

## Experiments and Evaluation
### Dataset Construction
- Use sampled labeled queries over your provided video corpus.
- Query buckets:
  - face-only
  - monument-only
  - combined face+monument

### Metrics
- Precision@K
- Recall@K
- MRR
- Success@K
- Secondary: indexing latency, query latency.

### Ablation Studies
- parser-only vs parser + keyword fallback
- strict-only vs strict + soft fallback
- confidence threshold sweep sensitivity

## Risks and Controls
- Face recognition uncertainty (`Maybe:` labels): preserve uncertainty and penalize score.
- Monument model is frame-level, not object-level: treat as frame evidence with explicit confidence.
- Missing labels in low-quality frames: soft fallback prevents zero-result failures.
- Timestamp precision tied to 1 fps extraction: frame filename `frame_NNNN.jpg` yields `timestamp_sec = N - 1` (e.g. `frame_0001.jpg` → 0 sec, `frame_0007.jpg` → 6 sec). Document this baseline assumption.

## Thesis Chapter Mapping
- Chapter 1: Problem and motivation (natural-language face/monument search in video)
- Chapter 2: Related work (video retrieval, face recognition, place recognition, metadata indexing)
- Chapter 3: System architecture and canonical schema
- Chapter 4: Query parsing + retrieval/ranking algorithm
- Chapter 5: Experimental setup and results (metrics + ablations)
- Chapter 6: Limitations and future work (vector retrieval, multilingual queries, temporal modeling)

## Implementation
- **Search service**: `search_engine.service` — `VisualSearchService` (index_video, parse_query, search, suggestions), label/timestamp helpers, strict/soft retrieval and ranking.
- **Web API**: `web.app` — `POST /api/index-video`, `GET /api/search`, `GET /api/search/suggestions`; lazy-instantiated `VisualSearchService`; `POST /api/process` supports `index_to_search`.
- **Tests**: `tests.test_visual_search_service` — label normalization, timestamp parsing, query slots, scoring.
- **Dependencies**: `pymongo` required for search APIs (see `requirements.txt`).

## Public Interface and Baseline Scope
- Baseline backend: MongoDB metadata/event retrieval.
- Baseline UI/API: indexing endpoint, search endpoint, suggestions endpoint.
- Out of baseline scope: vector DB reranking, generic object-color retrieval, multimodal text embeddings.
