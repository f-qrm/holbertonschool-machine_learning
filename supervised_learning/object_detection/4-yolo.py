#!/usr/bin/env python3
"""Contains the Yolo class for object detection"""

import cv2
import glob
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
        # Sauvegarder les seuils et les ancres
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
        # Charger le modèle Keras depuis le fichier .h5
        self.model = tf.keras.models.load_model(model_path)
        # Lire la liste des noms de classes (une par ligne)
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
        # Initialiser les listes de résultats
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            # Extraire et appliquer sigmoid sur le score de confiance (index 4)
            confidence = output[..., 4:5]
            confidence = tf.sigmoid(confidence).numpy()
            box_confidences.append(confidence)

            # Extraire et appliquer sigmoid sur les probabilités de classe
            class_probs = output[..., 5:]
            class_probs = tf.sigmoid(class_probs).numpy()
            box_class_probs.append(class_probs)

            # Extraire les 4 valeurs brutes de la boîte (tx, ty, tw, th)
            raw_boxes = output[..., 0:4]
            grid_height = output.shape[0]
            grid_width = output.shape[1]

            # Créer la grille des offsets de cellules (cx, cy)
            cx, cy = np.meshgrid(np.arange(grid_width), np.arange(grid_height))
            cx = cx.reshape(grid_height, grid_width, 1, 1)
            cy = cy.reshape(grid_height, grid_width, 1, 1)

            # Calculer le centre de la boîte: sigmoid(tx)+offset de la cellule
            t_x = raw_boxes[..., 0:1]
            t_y = raw_boxes[..., 1:2]
            bx = tf.sigmoid(t_x) + cx
            by = tf.sigmoid(t_y) + cy

            # Récupérer dimensions des ancres(largeur et hauteur prédéfinies
            pw = self.anchors[i][:, 0].reshape(1, 1, -1, 1)
            ph = self.anchors[i][:, 1].reshape(1, 1, -1, 1)

            # Calculer la taille de la boîte : ancre * exp(tw)
            t_w = raw_boxes[..., 2:3]
            t_h = raw_boxes[..., 3:4]
            bw = pw * np.exp(t_w)
            bh = ph * np.exp(t_h)

            # Taille d'entrée du modèle pour la normalisation
            input_width = self.model.input.shape[1]
            input_height = self.model.input.shape[2]

            # Convertir les coordonnées en pixels dans l'image originale
            bx_pixels = bx / grid_width * image_size[1]
            by_pixels = by / grid_height * image_size[0]
            bw_pixels = bw / input_width * image_size[1]
            bh_pixels = bh / input_height * image_size[0]

            # Convertir centre+taille en coins(x1,y1)haut-gauche
            # et(x2,y2)bas-droite
            x1 = bx_pixels - bw_pixels / 2
            y1 = by_pixels - bh_pixels / 2
            x2 = bx_pixels + bw_pixels / 2
            y2 = by_pixels + bh_pixels / 2

            # Assembler les coins en un tableau et l'ajouter à la liste
            box = np.concatenate([x1, y1, x2, y2], axis=-1)
            boxes.append(box)
        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filtre les boîtes selon le seuil de score de classe.

        Le score d'une boîte pour une classe donnée est :
            score = confidence * class_probability
        Seules les boîtes dont le meilleur score dépasse class_t sont gardées.

        Args:
            boxes (list of numpy.ndarray): boîtes traitées par process_outputs,
                chaque élément a la forme (grid_h, grid_w, anchor_boxes, 4)
            box_confidences (list of numpy.ndarray): scores de confiance,
                chaque élément a la forme (grid_h, grid_w, anchor_boxes, 1)
            box_class_probs (list of numpy.ndarray): probabilités de classe,
                chaque élément a la forme (grid_h, grid_w, anchor_boxes,
                num_classes)

        Returns:
            tuple: (filtered_boxes, box_classes, box_scores)
                - filtered_boxes (numpy.ndarray): boîtes retenues, forme
                (N, 4)
                - box_classes (numpy.ndarray): indice de la classe la plus
                probable pour chaque boîte retenue, forme (N,)
                - box_scores (numpy.ndarray): score correspondant à cette
                classe, forme (N,)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            # Score par classe=confiance de la boîte×probabilité de la classe
            scores = box_confidences[i] * box_class_probs[i]

            # Meilleur score et classe associée pour chaque boîte
            max_score = np.max(scores, axis=-1)
            max_score_class = np.argmax(scores, axis=-1)

            # Masque : on ne garde que les boîtes au-dessus du seuil class_t
            mask = max_score > self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(max_score_class[mask])
            box_scores.append(max_score[mask])

        # Fusionner les résultats de toutes les sorties en un seul tableau
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)
        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Apply Non-Max Suppression to eliminate redundant overlapping boxes.

        For each class, keeps only the highest-scoring box and removes any
        other box whose IoU with it exceeds nms_t. Repeats greedily until
        no boxes remain for that class.

        Args:
            filtered_boxes (numpy.ndarray): filtered boundary boxes from
                filter_boxes, shape (N, 4)
            box_classes (numpy.ndarray): class index for each filtered box,
                shape (N,)
            box_scores (numpy.ndarray): score for each filtered box,
                shape (N,)

        Returns:
            tuple: (box_predictions, predicted_box_classes,
                predicted_box_scores)
                - box_predictions (numpy.ndarray): predicted bounding boxes
                  ordered by class and score, shape (M, 4)
                - predicted_box_classes (numpy.ndarray): class index for each
                  predicted box, shape (M,)
                - predicted_box_scores (numpy.ndarray): score for each
                  predicted box, shape (M,)
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        # Traiter chaque classe indépendamment
        for c in np.unique(box_classes):
            mask = box_classes == c
            classe_boxes = filtered_boxes[mask]
            class_scores = box_scores[mask]

            # Trier les boîtes par score décroissant
            sorted_indices = np.argsort(class_scores)[::-1]
            classe_boxes = classe_boxes[sorted_indices]
            class_scores = class_scores[sorted_indices]

            # Sélection greedy : garder la meilleure boîte,
            # supprimer celles qui chevauchent trop
            while len(classe_boxes) > 0:
                # La boîte avec le meilleur score est toujours conservée
                box_predictions.append(classe_boxes[0])
                predicted_box_classes.append(c)
                predicted_box_scores.append(class_scores[0])
                if len(classe_boxes) == 1:
                    break

                # Calculer l'intersection entre la meilleure boîte
                # et les autres
                inter_x1 = np.maximum(
                    classe_boxes[0, 0], classe_boxes[1:, 0])
                inter_y1 = np.maximum(
                    classe_boxes[0, 1], classe_boxes[1:, 1])
                inter_x2 = np.minimum(
                    classe_boxes[0, 2], classe_boxes[1:, 2])
                inter_y2 = np.minimum(
                    classe_boxes[0, 3], classe_boxes[1:, 3])

                # Aire de l'intersection (0 si pas de chevauchement)
                length = np.maximum(inter_x2 - inter_x1, 0)
                wheight = np.maximum(inter_y2 - inter_y1, 0)
                inter_area = length * wheight

                # Aires individuelles des boîtes
                first_box_area = (
                    classe_boxes[0, 2] - classe_boxes[0, 0]) * (
                        classe_boxes[0, 3] - classe_boxes[0, 1])
                other_boxes_area = (
                    classe_boxes[1:, 2] - classe_boxes[1:, 0]) * (
                        classe_boxes[1:, 3] - classe_boxes[1:, 1])

                # IoU = intersection / union
                union = first_box_area + other_boxes_area - inter_area
                iou = inter_area / union

                # Ne garder que les boîtes dont l'IoU est sous le seuil nms_t
                keep = iou <= self.nms_t
                classe_boxes = classe_boxes[1:][keep]
                class_scores = class_scores[1:][keep]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """Load all images from a folder.

        Args:
            folder_path (str): path to the folder containing images

        Returns:
            tuple: (images, image_paths)
                - images (list of numpy.ndarray): loaded images
                - image_paths (list of str): corresponding file paths
        """
        image_paths = glob.glob(folder_path + '/*')
        images = [cv2.imread(path) for path in image_paths]
        return images, image_paths
