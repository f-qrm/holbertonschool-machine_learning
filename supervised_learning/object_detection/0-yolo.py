#!/usr/bin/env python3
"""Contains the Yolo class for object detection"""


import tensorflow as tf


class Yolo:
    """Uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initializes the Yolo model with its parameters"""
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
