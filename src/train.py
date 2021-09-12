from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedShuffleSplit

def train_test_split(n_split):
    """

    Function to return train and test sets
    :param : loaded images and labels with the number of splits
    :return: Train and test sets along with labels

    """
    skf = StratifiedShuffleSplit(n_splits=n_split, random_state=None, test_size=0.2)

    return skf


def train_model(model, X_train, y_train, batch_size, num_epochs, X_test, y_test, model_name):
    """

        Function to return trained model
        :param X_train:
        :param y_train:
        :param batch_size:
        :param num_epochs:
        :param y_test:
        :param model_name:
        :param model: Model to be trained, X_train:Training data, y_train:Training labels,
        batch_size:batch_size, num_epoch: number of epochs, X_test:Testing data, y_test: Testing labels
        :return: Trained model

    """

    filepath = "./models/" + model_name+"_model_weights.h5"
    checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
    callbacks_list = [checkpoint]

    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs,
                     verbose=1,
                     validation_data=(X_test, y_test), callbacks=callbacks_list)

    (loss, accuracy) = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

    return model, hist, loss, accuracy
