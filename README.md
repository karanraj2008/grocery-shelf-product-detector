# Grocery Shelf Product Detector

Detects individual products on grocery shelf images and counts them, using a custom object detector trained from scratch on the Grocery Dataset. The trained model is wrapped into a small Streamlit app so it can be used interactively, not just from inside a notebook.

## Project Overview

- Built a MobileNetV2-backbone, multi-scale object detector (3 detection heads, one per feature map scale) for locating products on shelf images
- The raw annotation file has no column headers, so the box format (`x, y, w, h, b`) was verified empirically rather than assumed - see `notebooks/annotation_check.ipynb`
- Trained and evaluated end-to-end in `notebooks/product_detection.ipynb`, achieving **[X] mAP@0.5** on the held-out test set
- Extracted the inference-only code into a standalone module (`app/detector.py`) and built a Streamlit app on top of it, so the model can be used outside the notebook by just uploading an image

## Project Structure

```
grocery-shelf-product-detector/
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── product_detection.ipynb        # model training + evaluation
│   └── annotation_check.ipynb         # annotation format verification
│
└── app/
    ├── detector.py                    # inference-only model wrapper
    ├── app.py                         # Streamlit UI
    ├── models/
    │   └── model_best.pth             # trained weights
    └── config_files/
        └── config.json                # img_size, strides, grid_sizes, anchors
```

## Data Sources

- **Annotations**: [gulvarol/grocerydataset](https://github.com/gulvarol/grocerydataset)
- **Shelf Images**: [ShelfImages on Kaggle](https://www.kaggle.com/datasets/sneha36h/selfimages4)

The dataset itself isn't committed to this repo (research-only license, and it's not needed to run the app - only to retrain). Download it from the links above and place it under `dataset/` if you want to run the training notebook.

## Installation

1. (Recommended) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

GPU is not compulsory but training will be much faster with it.

## Using the App

The trained model and its config are already included under `app/models/` and `app/config_files/`, so the app runs out of the box - no training required.

```bash
cd app
streamlit run app.py
```

This opens a page in your browser. Upload a shelf photo, and the app draws a box around every product it detects, along with a count.

## Training & Evaluation

See `notebooks/product_detection.ipynb` for the full pipeline: data loading, anchor computation, model architecture, training loop, and mAP evaluation on the test set.

## Annotation Format Verification

`annotation.txt` has no column headers, so before trusting it for training, `notebooks/annotation_check.ipynb` verifies the box format two ways:
- **Bounds check** - draws each box using both possible orderings (`x,y,w,h` and `x,y,h,w`) and checks which one keeps boxes inside the actual image dimensions
- **Visual check** - overlays both interpretations on a real shelf image side by side

Both checks agree on `x, y, w, h, b`, which is what's used for training.

## Model → App Integration

`app/detector.py` contains only what's needed for inference - the model architecture, the decode/NMS logic, and a `ProductDetectorRuntime` class with a simple `predict()` / `predict_and_draw()` interface. It's a deliberate subset of the notebook code: training loop, data augmentation, loss functions, and evaluation code were all left behind since they aren't needed once the model is trained.

`app/app.py` is a thin Streamlit UI on top of that - it just handles the file upload and displays the result.

## Results

| Metric | Score |
|---|---|
| mAP@0.5 | [X] |
| Precision | [X] |
| Recall | [X] |

(See `notebooks/product_detection.ipynb` for the PR curve and training loss plot.)
