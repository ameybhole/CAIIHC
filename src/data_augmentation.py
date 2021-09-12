import albumentations as A
import numpy as np


class Augmentation(object):

    """

    Class to data augmentation

    """

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

        """

        Function to return an object with augmentation transformations

        :param image: Image which will be transformed
        :return: Object with transformations

        """

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

        """

        Function to return augmented image

        :param image: image to be augmented
        :return: Augmented image


        """

        data = {"image": image}
        augmented_data = self.transformations(data)
        augmented_image = augmented_data["image"]

        return augmented_image

    def process_data(self, images):

        """

        Function to return an list of augmented images

        :param images: list of images to be augmented
        :return: Lists of augmented images

        """

        augmented_dataset = []

        for i in range(len(images)):
            augmented_image = self.augmentation_function(images[i])
            augmented_dataset.append(augmented_image)

        return np.array(augmented_dataset)
