# Thesis claims vs implementation (Chapters 4–8)

Legend: **Implemented** | **Partial** | **Not implemented** (in `vista-prototype` repo)

## Chapter 4 – Architecture / Vista-prototype

| Claim (as commonly stated in draft) | Status | Evidence |
|--------|--------|----------|
| Video upload as primary ingestion | **Partial** | Web UI uses **YouTube URL** + optional time window (`scan_start_seconds` / `scan_end_seconds` in `web/app.py`); not generic file upload for main processing. |
| Local file + URL processing | **Partial** | **`implementation.py`**: local `--video` or `--url` (YOLO + frames only). **Full** multimodal path is **`POST /api/process`** (web). |
| Unified pipeline always includes faces + monuments | **Partial** | Faces/monuments + MongoDB indexing on **web** path only. CLI entrypoint is **object detection only**. |
| Parallel per-frame object + face + monument | **Implemented** | `web/app.py` → YOLO, `run_face_detection`, `run_monument_recognition`, `save_detection_results`. |
| JSON / metadata outputs | **Implemented** | `detection_results.json`, `metadata.txt`, annotated frames, rendered video under `vista-prototype/results/<video_id>/`. |
| Example JSON with `faces[].name`, `monuments[]` | **Not implemented** | Actual JSON uses **`label`** on faces, singular **`monument`** object per frame, **`detections`** for YOLO (see `pipeline/detection.py` / `save_detection_results`). |
| MongoDB enables “retrieval” in prototype | **Partial** | **Indexing** only: `pipeline/mongodb_store.py` → `videos` + `frames` when `MONGODB_URI` set. No query API in app. |
| Schema: separate `detections` collection | **Not implemented** | **Nested** `objects`, `faces`, `monument` inside each **`frames`** document. |
| Web API: “retrieve metadata” as search | **Not implemented** | APIs: process, training, static `/results/...`; **no** `/api/search**. |

## Chapter 5 – Models / database

| Claim | Status | Evidence |
|--------|--------|----------|
| Face embeddings + monument classifier | **Implemented** | `scripts/build_models.py`, `known_faces/`, `monument_model/`, web `/training`. |
| Training folder layout | **Partial** | Thesis examples should match **`vista-prototype/training_data/faces/`** and **`.../monuments/`** (see README). |
| “Search database” fully exercised by queries | **Partial** | Data is **indexed**; **no** implemented search service in repo. |

## Chapter 6 – Visual Search Engine

| Claim | Status | Evidence |
|--------|--------|----------|
| Implemented query parsing module | **Not implemented** | No NL parser in `web/` or `pipeline/`. |
| Implemented retrieval engine + ranking | **Not implemented** | No `GET /api/search` or equivalent. |
| Strict / soft retrieval in code | **Not implemented** | Design-only in thesis draft; align to **future work** or conceptual section. |
| Example MongoDB filter `faces.name`, `monuments.label` | **Not implemented** | Store uses **`faces` array with `label`**, **`monument.label`**. |

## Chapter 7 – Implementation / integration

| Claim | Status | Evidence |
|--------|--------|----------|
| Flask app + process pipeline | **Implemented** | `web/app.py`. |
| Integration of prototype + “search engine” | **Partial** | **Processing + optional MongoDB write**; **not** a separate deployed search service. |
| `implementation.py` as full system | **Not implemented** | **YOLO-only**; no faces/monuments/MongoDB in that entrypoint. |

## Chapter 8 – Evaluation

| Claim | Status | Evidence |
|--------|--------|----------|
| Precision@K, Recall@K, MRR, Success@K **results** | **Not implemented** | No eval scripts, no qrels in repo. |
| Baseline vs proposed quantitative comparison | **Not implemented** | No logged experiments. |
| Ablation (strict vs soft, parser vs no parser) | **Not implemented** | No corresponding runs. |
| Indexing / query latency **measurements** | **Partial** | **`run_stats`** timing fields exist in web pipeline JSON when a run completes; not a formal Ch 8 study unless collected. |

---

This inventory drove edits to `Vista-Thesis.docx`, `Chapter-01-Revised.md`, and `docs/VISTA_VISUAL_SEARCH_ALGORITHM_SPEC.md`.

**Automation:** Re-run `python thesis/align_thesis_to_implementation.py` from the `thesis/` directory after editing `Chapter-01-Revised.md` or `alignment_chapters.py` (Chapter 6 title must be `Chapter 6: Conceptual Design for Retrieval and Ranking` after the first run).
