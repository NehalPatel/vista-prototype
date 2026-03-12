### Refactor Face Embedding Storage to Use One Mean Embedding Per Person

**Objective**

The current face recognition training pipeline stores **one `.npy` embedding per detected face image**.
This results in hundreds or thousands of `.npy` files and causes slow recognition because every detected face in video must be compared with all embeddings.

Refactor the system so that **each person has only one representative embedding**.

---

## Current Behaviour

Current pipeline:

1. Detect face from image
2. Generate embedding vector
3. Save embedding as a `.npy` file

Example directory structure:

```
embeddings/
   Tom_Cruise/
       0.npy
       1.npy
       2.npy
       3.npy
   Shahrukh_Khan/
       0.npy
       1.npy
       2.npy
```

During recognition:

```
detected_face_embedding
    ↓
compare with every .npy file
```

This becomes slow when many images exist.

---

# Required Refactor

Instead of storing **many embeddings per person**, compute a **single mean embedding per person**.

### Step 1 — Load All Embeddings of a Person

Read all `.npy` files inside each person's folder.

Example:

```
Tom_Cruise/
   0.npy
   1.npy
   2.npy
   ...
```

Load all embeddings into a list.

---

### Step 2 — Compute Mean Embedding

Convert list to numpy array and compute the mean.

```
mean_embedding = np.mean(embeddings, axis=0)
```

This creates one representative vector for the person.

---

### Step 3 — Store Database in One File

Create a dictionary structure:

```
{
   "Tom Cruise": embedding_vector,
   "Shahrukh Khan": embedding_vector,
   "Deepika Padukone": embedding_vector
}
```

Save it as:

```
face_database.npy
```

or

```
face_database.npz
```

---

### Step 4 — Recognition Pipeline Change

Instead of loading thousands of `.npy` files, load the database once:

```
database = np.load("face_database.npy", allow_pickle=True).item()
```

During recognition:

```
detected_face_embedding
     ↓
compare with each person's mean embedding
```

This reduces comparisons dramatically.

Example:

Before:

```
100 persons
20 images each
= 2000 comparisons
```

After:

```
100 comparisons
```

---

### Step 5 — Distance Calculation

Use cosine similarity or L2 distance.

Example:

```
distance = np.linalg.norm(embedding - person_embedding)
```

Apply threshold:

```
if distance < threshold:
    recognized
else:
    unknown
```

---

### Step 6 — Result

Benefits of this change:

- significantly faster recognition
- smaller storage footprint
- simpler database structure
- easier deployment
- scalable to large datasets

---

### Optional Future Optimization

For large databases (>1000 people), integrate a vector index such as FAISS for fast nearest-neighbour search.
