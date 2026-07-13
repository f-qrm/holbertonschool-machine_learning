#!/usr/bin/env python3
"""Contains the Yolo class for object detection"""


import numpy as np
import tensorflow as tf


class Yolo:
    """Uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize the Yolo model.

        Args:
            model_path (str): path to the Darknet Keras model
            classes_path (str): path to the file containing class names
            class_t (float): box score threshold for initial filtering
            nms_t (float): IOU threshold for non-max suppression
            anchors (numpy.ndarray): anchor boxes, shape (outputs,
            anchor_boxes, 2)
        """
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

    def process_outputs(self, outputs, image_size):
        """Process the outputs of the Darknet model for a single image.

        Args:
            outputs (list of numpy.ndarray): predictions from the
            Darknet model, each output has shape (grid_h, grid_w,
            anchor_boxes, 5 + num_classes) image_size (numpy.ndarray):
            original image size as [image_height, image_width]

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
                - boxes: list of numpy.ndarray of shape (grid_h, grid_w,
                anchor_boxes, 4) with processed boundary boxes for each output,
                as [x1, y1, x2, y2] in original image coordinates
                - box_confidences: list of numpy.ndarray of shape
                  (grid_h, grid_w, anchor_boxes, 1) with box confidence scores
                - box_class_probs: list of numpy.ndarray of shape
                  (grid_h, grid_w, anchor_boxes, num_classes) with class
                  probabilities
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            confidence = output[..., 4:5]
            confidence = tf.sigmoid(confidence).numpy()
            box_confidences.append(confidence)

            class_probs = output[..., 5:]
            class_probs = tf.sigmoid(class_probs).numpy()
            box_class_probs.append(class_probs)

            raw_boxes = output[..., 0:4]
            grid_height = output.shape[0]
            grid_width = output.shape[1]

            cx, cy = np.meshgrid(np.arange(grid_width), np.arange(grid_height))
            cx = cx.reshape(grid_height, grid_width, 1, 1)
            cy = cy.reshape(grid_height, grid_width, 1, 1)

            t_x = raw_boxes[..., 0:1]
            t_y = raw_boxes[..., 1:2]
            bx = tf.sigmoid(t_x) + cx
            by = tf.sigmoid(t_y) + cy

            pw = self.anchors[i][:, 0].reshape(1, 1, -1, 1)
            ph = self.anchors[i][:, 1].reshape(1, 1, -1, 1)

            t_w = raw_boxes[..., 2:3]
            t_h = raw_boxes[..., 3:4]
            bw = pw * np.exp(t_w)
            bh = ph * np.exp(t_h)

            input_width = self.model.input.shape[1]
            input_height = self.model.input.shape[2]

            bx_pixels = bx / grid_width * image_size[1]
            by_pixels = by / grid_height * image_size[0]
            bw_pixels = bw / input_width * image_size[1]
            bh_pixels = bh / input_height * image_size[0]

            x1 = bx_pixels - bw_pixels / 2
            y1 = by_pixels - bh_pixels / 2
            x2 = bx_pixels + bw_pixels / 2
            y2 = by_pixels + bh_pixels / 2

            box = np.concatenate([x1, y1, x2, y2], axis=-1)
            boxes.append(box)
        return boxes, box_confidences, box_class_probs
