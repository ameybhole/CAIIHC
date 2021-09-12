import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing


def printWrongPredictions(pred, y_test, labels):
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


def plot_data_graph(hist, num_epochs, model_name):
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['categorical_accuracy']
    val_acc = hist.history['val_categorical_accuracy']
    xc = range(num_epochs)

    plt.figure(figsize=(12, 10))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train', 'val'])
    plt.show()
    plt.savefig("./plots/" + model_name + "_loss")

    plt.figure(figsize=(12, 10))
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train', 'val'])
    plt.show()
    plt.savefig("./plots/" + model_name + "_acc")


def model_predictions(model, X_test):
    pred = model.predict(X_test)

    return pred
