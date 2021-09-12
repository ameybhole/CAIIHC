from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedShuffleSplit


def train_test_split(n_split):

    """

    Function to return train and test sets

    :param n_split : number of splits
    :return: Object for splitting dataset in a stratified way

    """

    skf = StratifiedShuffleSplit(n_splits=n_split, random_state=None, test_size=0.2)

    return skf


def train_model(model, X_train, y_train, batch_size, num_epochs, X_test, y_test, model_name):

    """

    Function to Train model

    :param X_test: Testing data
    :param X_train: Training data
    :param y_train: Training labels
    :param batch_size: Batch size
    :param num_epochs: Number of epochs
    :param y_test: Testing labels
    :param model_name: Name of model
    :param model: Model to be trained
    :return: Trained model, History of training, Loss, Accuracy

    """

    filepath = "./models/" + model_name + "_model_weights.h5"
    checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
    callbacks_list = [checkpoint]

    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs,
                     verbose=1,
                     validation_data=(X_test, y_test), callbacks=callbacks_list)

    (loss, accuracy) = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

    return model, hist, loss, accuracy
