import logging
import random
import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
from typing import List, Dict, Union
from imgaug import parameters as iap
from imgaug import dtypes as iadt
from imgaug import augmenters as iaa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Augmentation:
    """
    Class for data augmentation techniques.

    ...

    Attributes
    ----------
    augmenters : list
        List of imgaug augmenters to be applied.
    seeds : list
        List of random seeds for reproducibility.
    ...

    Methods
    -------
    add_aug(self, aug, seed=None):
        Add an augmentation technique to the pipeline.

    apply_aug(self, image, bboxes=None, labels=None):
        Apply the defined augmentation pipeline to the image and annotations.

    reset(self):
        Reset the augmentation pipeline.
    """

    def __init__(self):
        self.augmenters = []
        self.seeds = []

    def add_aug(self, aug: iaa.Augmenter, seed: int = None):
        """
        Add an augmentation technique to the pipeline.

        Parameters
        ----------
        aug : imgaug Augmenter
            Augmentation technique to be applied.
        seed : int, optional
            Random seed for reproducibility, by default None.
        """
        self.augmenters.append(aug)
        if seed:
            self.seeds.append(seed)

    def apply_aug(self, image: np.ndarray, bboxes: List[List[int]] = None, labels: List[str] = None) -> Dict[Union[str, np.ndarray]]:
        """
        Apply the defined augmentation pipeline to the image and annotations.

        Parameters
        ----------
        image : np.ndarray
            Input image to be augmented.
        bboxes : list of list of int, optional
            List of bounding boxes [[x1, y1, x2, y2], ...], by default None.
        labels : list of str, optional
            List of labels for each bounding box, by default None.

        Returns
        -------
        dict
            A dictionary containing the augmented image and the augmented annotations (if provided).
        """
        seq = iaa.Sequential(self.augmenters, random_order=True, random_state=iaa.RNG(self.seeds))
        image_aug = seq(image=image)

        if bboxes is not None and labels is not None:
            bboxes = np.array(bboxes)
            bboxes = seq.augment_bounding_boxes([iap.BoundingBox(x1, y1, x2, y2) for (x1, y1, x2, y2) in bboxes])
            bboxes = np.array([[bb.x1, bb.y1, bb.x2, bb.y2] for bb in bboxes])
            return {"image": image_aug, "bboxes": bboxes.tolist(), "labels": labels}
        else:
            return {"image": image_aug}

    def reset(self):
        """
        Reset the augmentation pipeline.
        """
        self.augmenters = []
        self.seeds = []

# Example usage
if __name__ == "__main__":
    aug = Augmentation()
    image = cv2.imread("example.jpg")
    bboxes = [[10, 10, 100, 100], [200, 300, 400, 500]]
    labels = ["car", "person"]

    # Add augmentation techniques
    aug.add_aug(iaa.GaussianBlur(sigma=(0, 3.0)), seed=42)
    aug.add_aug(iaa.Add((-40, 40)), seed=123)

    # Apply augmentation
    augmented_data = aug.apply_aug(image, bboxes, labels)
    augmented_image = augmented_data["image"]
    augmented_bboxes = augmented_data["bboxes"]
    augmented_labels = augmented_data["labels"]

    # Perform further processing or save augmented data
    ...

# Unit tests
import unittest

class TestAugmentation(unittest.TestCase):
    def setUp(self):
        self.aug = Augmentation()
        self.image = cv2.imread("test_image.jpg")
        self.bboxes = [[10, 10, 100, 100], [200, 300, 400, 500]]
        self.labels = ["car", "person"]

    def test_add_aug(self):
        aug = iaa.GaussianBlur(sigma=(0, 3.0))
        seed = 42
        self.aug.add_aug(aug, seed)
        self.assertEqual(len(self.aug.augmenters), 1)
        self.assertEqual(len(self.aug.seeds), 1)
        self.assertEqual(self.aug.seeds[0], seed)

    def test_apply_aug(self):
        # Add augmentation techniques for testing
        self.aug.add_aug(iaa.GaussianBlur(sigma=(0, 3.0)), seed=42)
        self.aug.add_aug(iaa.Add((-40, 40)), seed=123)

        # Apply augmentation and assert results
        augmented_data = self.aug.apply_aug(self.image, self.bboxes, self.labels)
        self.assertIn("image", augmented_data)
        self.assertIn("bboxes", augmented_data)
        self.assertIn("labels", augmented_data)
        self.assertEqual(len(augmented_data["bboxes"]), len(self.bboxes))
        self.assertEqual(len(augmented_data["labels"]), len(self.labels))

    def test_reset(self):
        # Add augmentation techniques
        self.aug.add_aug(iaa.GaussianBlur(sigma=(0, 3.0)), seed=42)
        self.aug.add_aug(iaa.Add((-40, 40)), seed=123)

        # Reset augmentation pipeline and assert empty lists
        self.aug.reset()
        self.assertEqual(len(self.aug.augmenters), 0)
        self.assertEqual(len(self.aug.seeds), 0)

if __name__ == "__main__":
    unittest.main()