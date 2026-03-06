Create a new page in the project called **"Training Data Manager"** where users can upload images to train the recognition system.

### Feature 1: Face Training

* Create a section called **"Face Training Dataset"**.
* Allow users to:

  * Enter the **Celebrity Name**.
  * Upload **multiple images** of that celebrity.
* Store the images in a structured format such as:

```
/training_data/faces/{celebrity_name}/image1.jpg
/training_data/faces/{celebrity_name}/image2.jpg
```

* The system will use these images to **train or register the face**.
* During **future video processing**, the system should:

  * Detect faces in video frames.
  * Compare them with the stored dataset.
  * If a match is found, display the **celebrity name**.

### Feature 2: Monument Recognition Training

Create another section called **"Monument Dataset"**.

Users should be able to:

* Enter the **Monument Name** (e.g., Taj Mahal, Eiffel Tower).
* Upload **multiple photos of the monument**.

Store them in a structure like:

```
/training_data/monuments/{monument_name}/image1.jpg
/training_data/monuments/{monument_name}/image2.jpg
```

The system should later:

* Detect monuments in images or video frames.
* Match them with the trained dataset.
* Display the **monument name when recognized**.

### UI Requirements

* Two tabs or sections:

  * **Face Training**
  * **Monument Training**
* Support **multiple file uploads**
* Show **preview of uploaded images**
* Include **Save / Train Dataset button**

### Optional Enhancements

* Show number of images uploaded per category
* Allow deleting images
* Show dataset list
