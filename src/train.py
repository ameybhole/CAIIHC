from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit

def train_test_split(n_split):

    """

    Function to return train and test sets
    :param : loaded images and labels with the number of splits
    :return: Train and test sets along with labels

    """
    skf = StratifiedShuffleSplit(n_splits=n_split, random_state=None, test_size=0.2)

    return skf

def train_model(model, X_train, y_train, batch_size, num_epochs, X_test, y_test):

    """

        Function to return trained model
        :param model: Model to be trained, X_train:Training data, y_train:Training labels,
        batch_size:batch_size, num_epoch: number of epochs, X_test:Testing data, y_test: Testing labels
        :return: Trained model

    """

    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs,
                                   verbose=1,
                                   validation_data=(X_test, y_test))

    (loss, accuracy) = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

    return model, hist, loss, accuracy
