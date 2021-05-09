"""Module for evaluating results of a Tensorflow model"""

from typing import Union

import cv2
import json
import numpy as np
from os import path

from tensorflow import lite


def _read_text_list(file_name: str) -> list[str]:
    with open(file_name) as text_file:
        return [line.strip() for line in text_file]


def _read_meta_data(file_name: str) -> dict:
    with open(file_name) as json_file:
        return json.load(json_file)


def _assert_is_classification_model(interpreter: lite.Interpreter):
    """Assert that the given TensorFlow model underneath the interpreter is a classification model
    """

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if not (len(input_details) == 1 and len(output_details) == 1):
        raise AssertionError("Tensor Model should be trained on one category (classification)")


class ClassificationModel:
    """Class for wrapping lower level TensorFlow model handling

    """

    _interpreter: lite.Interpreter
    _classifications: list[str]
    _meta_data: dict[str, Union[str, int, list[str]]]

    _input_details: dict
    _output_details: dict

    def __init__(self, model_path: str):
        self._interpreter = lite.Interpreter(path.join(model_path, 'model.tflite'))
        _assert_is_classification_model(self._interpreter)

        self._input_details = self._interpreter.get_input_details()[0]
        self._output_details = self._interpreter.get_output_details()[0]

        self._classifications = _read_text_list(path.join(model_path, 'dict.txt'))
        self._meta_data = _read_meta_data(path.join(model_path, 'tflite_metadata.json'))

        self._interpreter.allocate_tensors()

    def run(self, image: np.array) -> tuple[str, float]:
        """Return the result and confidence interval of running the image through
        the predictive classification model"""

        classification_predictor_index = self._input_details['index']
        classification_prediction_index = self._output_details['index']

        new_img = cv2.resize(image, self.get_image_dimension())
        test_img = np.expand_dims(new_img, axis=0)
        assert np.all(test_img.shape == self._input_details['shape'])
        breakpoint()

        self._interpreter.set_tensor(classification_predictor_index, test_img)
        self._interpreter.invoke()

        classification_scores = self._interpreter.get_tensor(classification_prediction_index)
        score_list = classification_scores[0]

        strongest_category = int(np.argmax(score_list))
        confidence = score_list[strongest_category] / np.sum(score_list)

        return self._classifications[strongest_category], confidence

    def run_from_file_path(self, file_path: str) -> tuple[str, float]:
        """Return self.run() but taking the file path of the image instead of the byte array"""
        image = cv2.imread(f"{file_path}")
        return self.run(image)

    def get_image_dimension(self) -> tuple[int, int]:
        """Return the image (width, height) the Classification Model is trained for"""
        return self._meta_data['imageWidth'], self._meta_data['imageHeight']
