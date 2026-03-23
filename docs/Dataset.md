Indian monuments dataset from Kaggle
https://www.kaggle.com/datasets/danushkumarv/indian-monuments-image-dataset

Indian cricketers dataset
https://www.kaggle.com/datasets/aviratgupta/indian-cricketers-dataset

Bollywood Celebrity Faces Localized Dataset (170)
https://www.kaggle.com/datasets/sroy93/bollywood-celeb-localized-face-dataset-extended

Bollywood celeb localized face dataset
https://www.kaggle.com/datasets/sushilyadav1998/bollywood-celeb-localized-face-dataset

## Organizing and building models

### Your current layout

- **Faces**: `training_data/datasets/faces/indian cricketer/<Cricketer Name>/` (324 persons).
- **Monuments**: `training_data/datasets/mounuments/Indian-monuments/images/train/<Name>/` and `.../test/<Name>/` (25 monument classes).

### Commands to organize data

From the **repo root** (`vista-prototype` project folder):

**1. Preview what will be copied (dry run):**
```bash
python scripts/organize_training_data.py --from-datasets --dry-run
```

**2. Organize both faces and monuments into `faces/` and `monuments/`:**
```bash
python scripts/organize_training_data.py --from-datasets
```

**3. Organize only faces:**
```bash
python scripts/organize_training_data.py --from-datasets --faces-only
```

**4. Organize only monuments:**
```bash
python scripts/organize_training_data.py --from-datasets --monuments-only
```

After organizing, `training_data/faces/` will have one folder per cricketer and `training_data/monuments/` one folder per monument (train+test merged).

### Build models (CLI)

By default, builds are **incremental**: only new or changed faces/images are processed; existing embeddings and monument features are reused. Face incremental state is stored in MongoDB in the `face_build_state` collection (same DB as detection results; set `MONGODB_URI` in `.env`). Paths are stored relative to the project so state stays valid across machines. Use `--full` for a from-scratch rebuild.

```bash
# Build both face and monument models (incremental)
python scripts/build_models.py

# From-scratch rebuild (clear existing face state and monument feature cache)
python scripts/build_models.py --full

# Only face model (from training_data/faces/)
python scripts/build_models.py --faces-only

# Only monument model (from training_data/monuments/ and training_data/dataset/)
python scripts/build_models.py --monuments-only
```

### Inbox (for future unorganized uploads)

- Put face images in `training_data/inbox_faces/<person_name>/`, then run `python scripts/organize_training_data.py` (no `--from-datasets`).
- Put monument images in `training_data/inbox_monuments/<monument_name>/`, then run the same.

All of these directories are in `.gitignore` so images are not pushed to GitHub.
