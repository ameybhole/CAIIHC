import os
import cv2
import numpy as np
from sklearn import preprocessing



def load_images(dataset):

    """

    Function to return an list of images and its labels
    :param dataset: Folder with class name and all the images in the dataset
    :return: Lists of image in an array form

    """

    img_data_list = []
    labels = []
    list_of_image_paths = []
    label_test = []

    data_dir_list = os.listdir(dataset)
    for folder_name in data_dir_list:
        img_list = os.listdir(dataset + '/' + folder_name)
        print(img_list)
        for image in img_list:
            retrieve_dir = dataset + "/" + folder_name + "/" + image
            images = cv2.imread(retrieve_dir, 3)
            images = cv2.resize(images, (224, 224))
            list_of_image_paths.append(images)
        img_data_list.append(img_list)
        labels.append(folder_name)
        label_test.append([folder_name] * len(img_list))

    return np.array(list_of_image_paths), label_test, labels

def binarize_labels(label_test):

    """

    Function to return an list of binarized labels
    :param dataset: list of labels
    :return: Lists of binarized labels

    """
    flattened_list = np.asarray([y for x in label_test for y in x], dtype="str")
    lb = preprocessing.LabelBinarizer().fit(flattened_list)
    labels = lb.transform(flattened_list)

    return labels






