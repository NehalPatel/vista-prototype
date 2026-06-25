# Urgent Thesis Fixes

These are the highest-priority fixes to apply across the full thesis before deeper chapter-by-chapter refinement.

## 1. Remove template language

Delete all instructional and note-style text such as:

- "Write This"
- "VERY IMPORTANT"
- "Add Later"
- "You MUST Add"
- emoji markers and checklist-style prompts

The thesis should read as a finished academic manuscript, not as a drafting guide.

## 2. Fix broken spacing and merged headings

There are multiple places where headings and sentences have collapsed together, for example:

- `Chapter 2: Literature ReviewReviews existing research...`
- `Single-Modality FocusMost systems...`
- `ScalabilityAbility to handle...`

These should be separated and reformatted consistently.

## 3. Replace outline fragments with formal prose

Many sections currently read like notes or slide bullets. Convert them into full paragraphs with transitions, justification, and a formal academic tone.

## 4. Add real citations

The literature review and technical choices need proper scholarly support. Add real references for:

- content-based video retrieval
- YOLO and object detection
- ArcFace / InsightFace / face embeddings
- landmark or monument recognition
- MongoDB / NoSQL metadata indexing for multimedia retrieval
- natural-language or semantic multimedia search

## 5. Distinguish research contribution from implementation

State clearly:

- what is the research contribution
- what is system engineering
- what is baseline design
- what is future work

This is especially important in Chapters 1, 2, 8, and 9.

## 6. Replace generic examples with actual system details

Use the implemented VISTA details consistently, including:

- frame extraction strategy
- actual JSON fields
- MongoDB collections and indexes
- strict and soft retrieval modes
- known-face and monument constraints

## 7. Strengthen evaluation rigor

Chapter 8 should include:

- dataset description
- query-set design
- baselines
- metrics
- result interpretation
- limitations and threats to validity

Raw scores without context should be replaced by proper tables and discussion.

## 8. Standardize terminology

Use these terms consistently:

- `VISTA` for the overall research system
- `Vista-prototype` for the video processing and indexing subsystem
- `Visual Search Engine` for the retrieval subsystem

## 9. Add real figures and tables

The thesis currently mentions diagrams and tables to add later. The final draft should actually include:

- system architecture diagram
- pipeline/data flow diagram
- schema diagram
- query parsing and retrieval flow
- sample result tables
- experimental result tables
- annotated frame and UI screenshots

## 10. Improve chapter endings and transitions

Each chapter should end with:

- a concise summary of what was established
- a bridge to the next chapter

This helps the thesis read as one coherent argument rather than a set of disconnected sections.
