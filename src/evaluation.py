import numpy as np
from sklearn import preprocessing


def printWrongPredictions(pred, y_test, labels):

    """

    Function to print list of incorrectly classified labels

    :param pred: Predictions
    :param y_test: Testing labels
    :param labels: List of labels
    :return: Incorreclty classified labels

    """

    array = []

    for i in enumerate(pred):
        biggest_value_index = pred[i].argmax(axis=0)
        value = pred[i][biggest_value_index]
        y_classes = pred[i] >= value
        y_classes = y_classes.astype(int)
        array.append(y_classes)
    predicted_list = np.asarray(array, dtype="int32")

    flattened_list = np.asarray([y for x in labels for y in x], dtype="str")
    lb = preprocessing.LabelBinarizer().fit(flattened_list)
    y_test = lb.inverse_transform(y_test)
    pred = lb.inverse_transform(predicted_list)

    for i in enumerate(y_test):
        if y_test[i] != pred[i]:
            print("Predicted:", pred[i], " Actual: ", y_test[i])


def model_predictions(model, X_test):

    """

    Function to predictions

    :param X_test: Testing data
    :param model: Model to  evaluate
    :return: Predictions

    """

    pred = model.predict(X_test)

    return pred
