"""Module for evaluating the classifications of an image"""

import pathlib
import argparse

import cv2

from workers import image_classification
from workers.evaluate import ClassificationModel

import sys
sys.argv = ["image_test_model.py", "models/larger_model", "assets/test_data"]

directory_parser = argparse.ArgumentParser()
directory_parser.add_argument("tfliteModelPath")
directory_parser.add_argument("testDataDirectory")

args = directory_parser.parse_args()

cv_images = []
for file in pathlib.Path(args.testDataDirectory).iterdir():
    if ".gif" not in str(file):
        img = cv2.imread(f"{file.resolve()}")
        cv_images.append((file, img))

classification_model = ClassificationModel(args.tfliteModelPath)
image_classification.evaluate_model(classification_model, cv_images)
