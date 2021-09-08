import albumentations as A
import numpy as np


class Augmentation(object):

    def __init__(
            self,
            rotationleft,
            rotationright,
            bright,
            dark
    ):
        self.rotationleft = rotationleft
        self.rotationright = rotationright
        self.bright = bright
        self.dark = dark


    def transformations(self, image):
        transforms = A.Compose([
            A.Rotate(limit=self.rotationleft),
            A.Rotate(limit=self.rotationright),
            A.RandomBrightness(limit=self.bright),
            A.RandomBrightness(limit=self.dark),
            A.HorizontalFlip(),
            A.GaussNoise()
        ])

        return transforms(**image)

    def augmentation_function(self, image):
        data = {"image": image}
        augmented_data = self.transformations(data)
        augmented_image = augmented_data["image"]

        return augmented_image

    def process_data(self, images):
        augmented_dataset = []

        for i in range(len(images)):
            augmented_image = self.augmentation_function(images[i])
            augmented_dataset.append(augmented_image)

        return np.array(augmented_dataset)
