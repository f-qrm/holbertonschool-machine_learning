# YOLOv3 Object Detection from Scratch

This project implements the full inference pipeline of the YOLOv3 object detector around a pretrained Darknet-based Keras model. Rather than calling a high-level detection API, it decodes the raw output tensors produced by the network's three detection heads, converts them into bounding boxes in image coordinates, filters them by confidence, removes redundant overlapping detections with non-max suppression, and finally draws the surviving boxes and class labels back onto the original images.

## Overview

Most practitioners interact with object detectors through a single `predict()` call and never see what happens in between. YOLOv3 does not output bounding boxes directly — each of its three output scales produces a grid of raw activations that must be transformed through sigmoid and exponential functions, combined with predefined anchor box shapes, and rescaled from the network's input resolution back to the original image size before they mean anything. Understanding this decoding step — anchor boxes, grid-cell offsets, objectness confidence, per-class scores, and Intersection-over-Union based suppression — is what separates someone who can call a detection library from someone who can debug, retrain, or adapt one to a different set of anchors, input resolution, or class list.

## Contents

| File | Description |
|------|-------------|
| `0-yolo.py` | `Yolo.__init__`: loads the pretrained Darknet/YOLOv3 Keras model (`yolo.h5`), reads class names from a text file, and stores the confidence threshold, NMS IoU threshold, and anchor boxes. |
| `1-yolo.py` | Adds `process_outputs`: decodes the raw model outputs (one per detection scale) into bounding box corners, box confidences, and class probabilities. |
| `2-yolo.py` | Adds `filter_boxes`: removes boxes whose best class score (`confidence * class_probability`) falls below `class_t`. |
| `3-yolo.py` | Adds `non_max_suppression`: performs greedy, per-class IoU-based suppression to eliminate duplicate detections of the same object. |
| `4-yolo.py` | Adds `load_images` (static method): loads every image from a folder with OpenCV. |
| `5-yolo.py` | Adds `preprocess_images`: resizes images to the model's input shape and normalizes pixel values to `[0, 1]`. |
| `6-yolo.py` | Adds `show_boxes`: draws bounding boxes and class/score labels on an image and optionally saves the result. |
| `7-yolo.py` | Adds `predict`: runs the complete pipeline end to end over a folder of images (load, preprocess, infer, decode, filter, suppress, display). |
| `0-main.py` … `7-main.py` | Example driver scripts exercising the corresponding `N-yolo.py` version (not deliverables). |
| `coco_classes.txt` | The 80 COCO class names, in the index order expected by the model's output. |
| `yolo.h5` | Pretrained Darknet-based YOLOv3 Keras model used for inference. |
| `yolo_images/` (and `.zip`) | Sample images (`dog.jpg`, `eagle.jpg`, `giraffe.jpg`, `horses.jpg`, `person.jpg`, `takagaki.jpg`) used to demonstrate the pipeline. |

## How It Works

The pipeline is implemented incrementally across `0-yolo.py` through `7-yolo.py`, but all stages live in a single `Yolo` class; `7-yolo.py` is the complete version described below.

**Model and class loading (`__init__`).** The constructor loads the pretrained model with `tf.keras.models.load_model(model_path)`, reads one class name per line from `classes_path`, and stores `class_t` (box score threshold), `nms_t` (NMS IoU threshold), and `anchors` — a `(3, 3, 2)` array of anchor box widths/heights, one triplet per detection scale.

**Decoding raw outputs (`process_outputs`).** YOLOv3 predicts at three grid resolutions. For each output tensor of shape `(grid_h, grid_w, anchor_boxes, 5 + num_classes)`:
- Objectness (`box_confidences`) and per-class scores (`box_class_probs`) are obtained by applying a sigmoid to channel 4 and channels 5 onward, respectively.
- Box centers are computed as `sigmoid(tx) + cx` and `sigmoid(ty) + cy`, where `(cx, cy)` are the grid-cell offsets generated with `np.meshgrid`, so a box's center is constrained to its own cell and expressed in grid units.
- Box dimensions are computed as `anchor_w * exp(tw)` and `anchor_h * exp(th)`, scaling each of the three anchor boxes assigned to that output.
- Centers are rescaled by the grid size and dimensions by the model's input size, then both are rescaled again to the original image's `[height, width]` (passed in as `image_size`), and converted from center/size to corner format `(x1, y1, x2, y2)`.

**Confidence-based filtering (`filter_boxes`).** For every box, a per-class score is computed as `box_confidence * class_probability`. Only the best class and its score are kept per box (`argmax`/`max` along the class axis), and boxes whose best score does not exceed `class_t` are discarded. The three output scales are then flattened and concatenated into single `(N, 4)` box, `(N,)` class, and `(N,)` score arrays.

**Non-max suppression (`non_max_suppression`).** Boxes are grouped by predicted class and processed independently, so overlapping detections of different classes never suppress each other. Within each class, boxes are sorted by descending score; the highest-scoring box is kept, and IoU is computed against all remaining boxes as `intersection_area / (area_a + area_b - intersection_area)` (with the intersection width/height clamped to 0 when the boxes do not overlap). Any box with IoU above `nms_t` relative to the kept box is discarded, and the process repeats greedily on what remains until each class's candidates are exhausted.

**Image loading and preprocessing (`load_images`, `preprocess_images`).** `load_images` globs every file in a folder and reads it with `cv2.imread`. `preprocess_images` resizes each image to the model's expected input width/height using cubic interpolation (`cv2.INTER_CUBIC`) and normalizes pixel values from `[0, 255]` to `[0, 1]`, while recording each image's original `[height, width]` so `process_outputs` can later rescale predictions back to that image's native resolution.

**Visualization (`show_boxes`) and full inference (`predict`).** `show_boxes` draws each predicted box in blue with `cv2.rectangle`, overlays the class name and rounded score in red with `cv2.putText`, and displays the image in a window; pressing `s` saves it to a `detections/` folder. `predict` ties everything together: it loads and preprocesses every image in a folder, runs a single batched forward pass with `self.model.predict`, then for each image individually re-applies `process_outputs`, `filter_boxes`, and `non_max_suppression` before calling `show_boxes`, and returns the list of final `(boxes, classes, scores)` predictions alongside the corresponding image paths.

## Requirements

- Python 3.12
- TensorFlow 2.21.0
- Keras 3.14.1
- NumPy 2.4.5
- opencv-python (`cv2`), used for image loading, resizing, drawing, and display

## Usage

```python
#!/usr/bin/env python3
import numpy as np
from yolo import Yolo  # 7-yolo.py

anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                     [[30, 61], [62, 45], [59, 119]],
                     [[10, 13], [16, 30], [33, 23]]])

yolo = Yolo('yolo.h5', 'coco_classes.txt', 0.6, 0.5, anchors)
predictions, image_paths = yolo.predict('yolo_images/yolo/')
```

`predict` opens a window per image with its predicted boxes drawn in blue and each label ("class_name score") in red above the box; pressing `s` saves the annotated image to `detections/`, any other key moves to the next image without saving. `predictions` is a list of `(boxes, classes, scores)` tuples, one per image, matching the order of `image_paths`.

## Design Notes

- Non-max suppression is applied per class rather than globally, so a correctly detected dog and a correctly detected bicycle standing in front of it are never suppressed against each other just because their boxes overlap.
- Preprocessing resizes with cubic interpolation directly to the network's fixed input shape and normalizes to `[0, 1]`, matching how the underlying Darknet model was trained, while the original image dimensions are carried alongside the batch so box coordinates can be mapped back to native resolution after inference.
- Box center and size decoding is kept separate from the corner-coordinate conversion (`sigmoid`/`exp` transforms first, pixel-space corners last), which keeps the anchor-box math easy to verify against the YOLOv3 paper independently of the final coordinate system.
- `predict` batches the forward pass across all images in a folder (`self.model.predict(pimages)`) but decodes and filters each image separately, since box counts differ per image and cannot be vectorized across a batch after NMS.

## Author

Fjolla Qerimi
