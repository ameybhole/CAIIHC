import numpy as np


from src.data_augmentation import Augmentation
from src.data_load import load_images, binarize_labels

from src.models import Models
from src.train import train_test_split, train_model

if __name__ == '__main__':
    list_of_image_paths, labels = load_images("D:\Amey\Masters\Projects\CAIIHC\data")
    binarizelabels = binarize_labels(labels)
    skf = train_test_split(5)
    model = Models(True, 136, 'imagenet', False, 'avg', 'softmax')
    model.resnet50()
    model.model_trainable()
    model_check = model.model_compile()
    for train_index, test_index in skf.split(list_of_image_paths, binarizelabels):
        X_train, X_test = list_of_image_paths[train_index], list_of_image_paths[test_index]
        y_train, y_test = binarizelabels[train_index], binarizelabels[test_index]
        augmentations = Augmentation(10, -10, 1.5, 0.5)
        augmented_data = augmentations.process_data(X_train)
        X_train_augmneted = np.concatenate((X_train, augmented_data), axis=0)
        y_train_augmented = np.concatenate((y_train, y_train), axis=0)
        train_model(model_check, X_train_augmneted, y_train_augmented , 32, 10, X_test, y_test)



