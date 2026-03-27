# -*- coding: utf-8 -*-
"""Replacement paragraph blocks (style name, text) for thesis alignment with vista-prototype."""

# Styles must exist in the document template (Heading 2, Heading 3, Normal).

CH4_BLOCKS = [
    ("Heading 2", "4.1 Introduction"),
    (
        "Normal",
        "This chapter describes the architecture of the Vista-prototype as implemented in the accompanying repository. The prototype has two entry paths: a Flask web application (primary) and a command-line script for object detection only. Optional MongoDB indexing persists frame-level evidence after a successful web run when environment variables are set.",
    ),
    (
        "Heading 2",
        "4.2 Web Application versus Command-Line Entry Points",
    ),
    (
        "Normal",
        "The main implementation path is the Flask app in web/app.py. Users supply a YouTube URL; the server downloads the video (PyTube with yt-dlp fallback), extracts frames over a configurable time window (scan_start_seconds and scan_end_seconds, defaulting to a short segment), runs YOLO-based object detection, optional InsightFace-based face detection and identity matching when known-face embeddings exist, and optional monument classification when a trained model is available. Results are written under vista-prototype/results/<video_id>/ and may be mirrored to MongoDB.",
    ),
    (
        "Normal",
        "The script implementation.py provides a lighter CLI path: it extracts frames and runs YOLO only. It does not run face recognition, monument classification, or MongoDB indexing. The thesis therefore distinguishes clearly between the full multimodal pipeline (web) and the object-detection-only CLI.",
    ),
    (
        "Heading 2",
        "4.3 Overall Processing Pipeline",
    ),
    (
        "Normal",
        "For the web path, the logical stages are: (1) video acquisition, (2) frame extraction, (3) object detection on each frame, (4) face detection and optional recognition, (5) monument classification per frame, (6) aggregation into detection_results.json, (7) rendering of an annotated video, (8) optional upsert into MongoDB collections videos and frames.",
    ),
    (
        "Heading 2",
        "4.4 Video Ingestion",
    ),
    (
        "Normal",
        "Ingestion is URL-centric in the web UI: the user pastes a YouTube watch URL. The pipeline resolves a video_id from the URL, downloads the highest-quality progressive MP4 when available, and stores the file under vista-prototype/videos/. The CLI accepts either --url or a local --video path; the same download helper is used for URLs.",
    ),
    (
        "Heading 2",
        "4.5 Frame Extraction Strategy",
    ),
    (
        "Normal",
        "Frames are extracted with OpenCV at a user-selected rate (frames per second) from the chosen time segment, not necessarily the entire file. Each saved frame file is named predictably (e.g., frame_0001.jpg) so that frame index maps to timestamp given the extraction FPS.",
    ),
    (
        "Heading 2",
        "4.6 Object Detection Module",
    ),
    (
        "Normal",
        "Object detection uses Ultralytics YOLOv8. Each detection includes class name, confidence, bounding box, and a dominant color label where computed, exposed in JSON as class, color, label, bbox, and conf. The selected weights (e.g., yolov8n) are recorded in the output JSON.",
    ),
    (
        "Heading 2",
        "4.7 Face Detection and Recognition Module",
    ),
    (
        "Normal",
        "Face detection uses InsightFace (Buffalo L/S/SC). When embeddings built via the training workflow exist under known_faces/, the pipeline matches detected faces to identities and attaches labels and recognition confidence. Without embeddings, faces may appear as generic detections or Unknown. Face results are stored per frame as a list of objects with label, confidence, recognition_confidence, and bbox.",
    ),
    (
        "Heading 2",
        "4.8 Monument Recognition Module",
    ),
    (
        "Normal",
        "Monument recognition uses a frame-level classifier trained from user-provided images (see Chapter 5). It assigns one label and confidence per frame, not a tight geographic bounding box around a landmark. For visualization, the implementation draws a full-frame highlight when confidence exceeds the threshold, and may store a pseudo-bounding box for UI consistency. This should be understood as frame-level evidence, not fine-grained landmark localization.",
    ),
    (
        "Heading 2",
        "4.9 Structured Outputs and JSON Representation",
    ),
    (
        "Normal",
        "Each processed video produces detection_results.json listing frames; each frame entry contains the frame filename, a detections array (YOLO), a faces array, and an optional monument object. Field names use label for recognized identities and monument labels, not a separate name field at the JSON level. metadata.txt records the source URL, counts, models, and device information.",
    ),
    (
        "Heading 2",
        "4.10 Optional MongoDB Indexing",
    ),
    (
        "Normal",
        "When MONGODB_URI or MONGO_URI is set, pipeline.mongodb_store.index_detection_results_to_mongodb upserts one document per video in the videos collection and replaces per-frame rows in the frames collection. Each frames document includes video_id, frame_filename, frame_index, time_sec, objects (array), faces (array with label), and monument (singular object with label and confidence). Indexes support faces.label, objects.class, objects.color, objects.label, and monument.label. This layer prepares data for search; it does not by itself expose a query HTTP API in the prototype.",
    ),
    (
        "Heading 2",
        "4.11 Web Interface and HTTP APIs",
    ),
    (
        "Normal",
        "The web UI serves static pages and JSON APIs: POST /api/process runs the pipeline for a YouTube URL; training-related routes support dataset upload and model builds under /training. Processed assets are served from /results/<video_id>/.... There is no implemented GET /api/search endpoint in the repository.",
    ),
    (
        "Heading 2",
        "4.12 Design Considerations",
    ),
    (
        "Normal",
        "Key engineering considerations include modular separation between download, detection, face, and monument steps; idempotent result directories per video_id; optional GPU use for torch-backed models; and graceful degradation when optional dependencies or models are missing.",
    ),
    (
        "Heading 2",
        "4.13 Summary",
    ),
    (
        "Normal",
        "This chapter aligned the architectural description with the implemented Vista-prototype: web-first multimodal processing, CLI object-only path, accurate JSON and MongoDB shapes, and explicit scope boundaries for retrieval. The next chapter details model training and how outputs populate the index.",
    ),
]

CH6_BLOCKS = [
    ("Heading 2", "6.1 Introduction"),
    (
        "Normal",
        "This chapter describes the intended Visual Search Engine as a conceptual layer on top of the indexed data produced by the Vista-prototype. Query parsing, strict and soft retrieval modes, and ranking are design targets for future implementation; they are not shipped as executable modules in the prototype repository. The discussion is grounded in the actual MongoDB and JSON schema from Chapter 4.",
    ),
    ("Heading 2", "6.2 Intended Modular Architecture"),
    (
        "Normal",
        "A future search service would logically comprise: query input, lightweight natural-language parsing (slot filling for person, monument, object class or color, optional time window), translation to MongoDB filters, execution against the frames collection, scoring and ranking, and grouping results by video_id with frame-level evidence.",
    ),
    ("Heading 2", "6.3 Illustrative Query Interpretation"),
    (
        "Normal",
        "For a query such as Nehal at Taj Mahal, a parser would extract a person slot and a monument slot. These map to indexed face labels and monument.label respectively. Attribute queries such as person in red car would additionally constrain objects by class and color fields nested in each frame document.",
    ),
    ("Heading 2", "6.4 Example MongoDB Filter Shapes (Illustrative)"),
    (
        "Normal",
        "Because faces is an array, person constraints use element matching, for example frames where faces contains an element with label matching the identity. Monument constraints reference monument.label. Object color and class use objects.color and objects.class. Time bounds use time_sec. These patterns match the fields written by pipeline.mongodb_store.",
    ),
    ("Heading 2", "6.5 Strict versus Soft Retrieval (Design)"),
    (
        "Normal",
        "Strict retrieval would require all active slots to match the same frame (logical AND). Soft retrieval could relax constraints or fall back to token-style matching on labels when strict mode returns no hits. Implementing and evaluating these modes is left to future work once a search API exists.",
    ),
    ("Heading 2", "6.6 Ranking (Design)"),
    (
        "Normal",
        "Ranking could combine recognition confidence, detection confidence, and slot coverage. The prototype already stores per-detection confidences suitable for such scoring but does not compute ranked result lists.",
    ),
    ("Heading 2", "6.7 Summary"),
    (
        "Normal",
        "Chapter 6 reframes the Visual Search Engine as specification and rationale tied to real indexed fields, without claiming a completed implementation. Chapter 7 maps the concrete codebase to the processing stages described so far.",
    ),
]

CH7_BLOCKS = [
    ("Heading 2", "7.1 Introduction"),
    (
        "Normal",
        "This chapter maps the Vista-prototype to source files in the repository. Implementation work centers on web/app.py, pipeline modules, face_pipeline for embeddings and matching, optional pymongo in pipeline.mongodb_store, and scripts such as build_models.py for training assets.",
    ),
    ("Heading 2", "7.2 Web Server and Process API"),
    (
        "Normal",
        "Flask routes in web/app.py implement the user interface, POST /api/process, training upload and train endpoints, and static serving of results. Processing orchestrates pipeline.video, pipeline.detection.run_yolo, pipeline.faces.run_face_detection, pipeline.monuments.run_monument_recognition, pipeline.render.make_video_from_images, and save_detection_results followed by optional index_detection_results_to_mongodb.",
    ),
    ("Heading 2", "7.3 Core Pipeline Modules"),
    (
        "Normal",
        "pipeline/detection.py implements YOLO inference and JSON serialization helpers. pipeline/faces.py integrates InsightFace detection and optional recognition. pipeline/monuments.py trains and applies the monument model. pipeline/paths.py centralizes runtime directories under vista-prototype/.",
    ),
    ("Heading 2", "7.4 Command-Line Entry Point"),
    (
        "Normal",
        "implementation.py wires only download or local file input, frame extraction, YOLO, metadata, and rendered video. Researchers extending the CLI should call the same helpers as the web app if parity with multimodal output is required.",
    ),
    ("Heading 2", "7.5 Persistence Layer"),
    (
        "Normal",
        "MongoDB access is isolated in pipeline/mongodb_store.py with environment-driven configuration. Failure to connect skips indexing without blocking local JSON output.",
    ),
    ("Heading 2", "7.6 Integration Scope"),
    (
        "Normal",
        "There is no separate deployable search microservice in the repository. Integration therefore means the processing stack plus optional database writes, not a second long-running retrieval service.",
    ),
    ("Heading 2", "7.7 Summary"),
    (
        "Normal",
        "The implementation chapter now reflects files and behaviors that exist in the codebase. Evaluation in Chapter 8 is limited accordingly to processing observations and planned retrieval studies.",
    ),
]

CH8_BLOCKS = [
    ("Heading 2", "8.1 Introduction"),
    (
        "Normal",
        "This chapter evaluates what the implemented prototype can support today: end-to-end processing quality, optional indexing, and timing fields captured in run_stats during web runs. It does not report Precision@K, Recall@K, MRR, or Success@K values, because no labeled query set or search service was executed in the repository. Those metrics remain appropriate goals for future work once retrieval is implemented and annotated.",
    ),
    ("Heading 2", "8.2 Experimental Environment"),
    (
        "Normal",
        "Experiments follow the hardware and software stack actually used for development: Python, Flask, OpenCV, Ultralytics YOLO, optional CUDA, optional InsightFace and onnxruntime, and optional pymongo. Exact machine specifications should be filled in by the author to match their thesis submission environment.",
    ),
    ("Heading 2", "8.3 Datasets and Test Videos"),
    (
        "Normal",
        "Evaluation is qualitative and operational: publicly suggested test videos from the project README, plus any custom YouTube URLs and locally trained face and monument datasets the author prepared. There is no fixed benchmark corpus bundled with the repository.",
    ),
    ("Heading 2", "8.4 Processing Metrics and Observations"),
    (
        "Normal",
        "For each completed web run, detection_results.json may include run_stats with download_sec, extract_frames_sec, detection_sec, face_detection_sec, monument_recognition_sec, render_sec, and total_sec, along with device and GPU name when available. These quantities support analysis of pipeline cost and bottlenecks. The author should cite measurements collected from their own runs rather than invented tables.",
    ),
    ("Heading 2", "8.5 Planned Information-Retrieval Evaluation"),
    (
        "Normal",
        "A rigorous IR study would require a frozen corpus, defined query topics, relevance judgments over frames or clips, baselines (e.g., keyword-only or single-modality filters), and reporting of Precision@K, Recall@K, MRR, and Success@K. Designing that protocol is encouraged; presenting numeric results without executing it would be misleading, so none are claimed here.",
    ),
    ("Heading 2", "8.6 Qualitative Assessment"),
    (
        "Normal",
        "Qualitative review should mention annotated frame outputs, correctness of YOLO labels at the chosen confidence threshold, stability of face recognition when embeddings exist, and monument predictions under the frame-level model limitations described in Chapter 4.",
    ),
    ("Heading 2", "8.7 Limitations Observed"),
    (
        "Normal",
        "Limitations include dependence on video quality, YouTube availability, partial segment scanning by default, optional components not installed in every environment, frame-level monument semantics, and the absence of an automated search UI.",
    ),
    ("Heading 2", "8.8 Summary"),
    (
        "Normal",
        "Chapter 8 now states honestly that evaluation emphasizes processing performance and qualitative behavior, while full retrieval evaluation awaits future implementation. Chapter 9 discusses implications under that scope.",
    ),
]

CH9_BLOCKS = [
    ("Heading 2", "9.1 Introduction"),
    (
        "Normal",
        "This chapter interprets the prototype in light of Chapters 4 through 8. The discussion focuses on multi-modal processing outcomes, indexing readiness, and the gap between persisted evidence and a user-facing search experience.",
    ),
    ("Heading 2", "9.2 Interpretation of Results"),
    (
        "Normal",
        "Object detection with YOLO provides a stable baseline for tagging frames with common categories. Face recognition quality depends on training coverage and thresholding. Monument classification offers useful coarse signals but should not be overstated as precise geographic detection. Together, these modalities produce a richer per-frame record than any single module alone.",
    ),
    ("Heading 2", "9.3 Design Trade-offs"),
    (
        "Normal",
        "Trade-offs include extraction FPS versus compute cost, confidence thresholds versus recall, optional MongoDB dependency versus file-only workflows, and web complexity versus the minimal CLI path.",
    ),
    ("Heading 2", "9.4 Limitations"),
    (
        "Normal",
        "Key limitations include no built-in search API, CLI parity gaps, URL-centric ingestion in the web UI, and lack of published IR metrics. External factors such as model licensing and dataset bias also apply.",
    ),
    ("Heading 2", "9.5 Threats to Validity"),
    (
        "Normal",
        "Without formal retrieval experiments, conclusions about search quality are speculative. Processing timings vary with hardware and network. YouTube-specific ingestion may not generalize to all hosted sources without additional adapters.",
    ),
    ("Heading 2", "9.6 Summary"),
    (
        "Normal",
        "The discussion aligns claims with the implemented Vista-prototype and clearly separates validated engineering outcomes from forward-looking retrieval goals.",
    ),
]

CH10_BLOCKS = [
    ("Heading 2", "10.1 Introduction"),
    (
        "Normal",
        "This thesis demonstrated a multimodal video understanding pipeline with optional MongoDB indexing suitable for future visual search. The closing chapter summarizes contributions candidly and lists research extensions.",
    ),
    ("Heading 2", "10.2 Contributions"),
    (
        "Normal",
        "Contributions include: integrated web pipeline for YOLO, faces, and monuments; structured JSON outputs; optional MongoDB videos and frames collections with practical indexes; training workflows for faces and monuments; and a conceptual retrieval design aligned with stored fields.",
    ),
    ("Heading 2", "10.3 Limitations"),
    (
        "Normal",
        "Limitations include missing search service, CLI scope mismatch, frame-level monument semantics, and absence of large-scale IR evaluation.",
    ),
    ("Heading 2", "10.4 Future Work"),
    (
        "Normal",
        "Future work should prioritize a search API (for example Flask or FastAPI routes) implementing parsing and MongoDB queries, a small labeled query set for IR metrics, unified CLI parity if desired, richer temporal modeling, and optional vector or hybrid retrieval once baselines are stable.",
    ),
    ("Heading 2", "10.5 Final Remarks"),
    (
        "Normal",
        "The Vista-prototype provides a concrete foundation for content-based video understanding and index construction; completing the retrieval layer remains the natural next research and engineering step.",
    ),
]
