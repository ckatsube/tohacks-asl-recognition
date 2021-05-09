"""Module for passing through a list of images through a classification model"""
import os

from workers.evaluate import ClassificationModel


def evaluate_model(model: ClassificationModel, images: list) -> None:
    """Evaluates a list of images against the model interpreter"""

    correct = []

    for file, img in images:
        file_name = os.path.basename(file)

        print("====================")
        print("Testing:", file_name)

        result, confidence = model.run(img)
        if confidence > 0.5:
            print("Classification result: ", result)
            if file_name[0].upper() == result or ("space" in file_name and result == "space"):
                print("Correct classification!")
                correct.append(1)
            else:
                print("Incorrect classification")
                correct.append(0)
        else:
            print("Classification result: no result")
            if "nothing" in file_name:
                print("Correct classification!")
                correct.append(1)
            else:
                print("Incorrect classification")
                correct.append(0)

    print("Test accuracy:", sum(correct) / len(correct))
