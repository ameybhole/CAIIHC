from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from skimage.color import rgb2gray
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def visualize_scatter_with_images(X_2d_data, images, figsize=(50, 50), image_zoom=0.2):

    """

    Function to plot scatter of images

    :param image_zoom: Image zoom
    :param figsize: Size of the figure
    :param images: List of images
    :param X_2d_data: 2D data
    :return: Lists of image in an array form

    """

    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(X_2d_data, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2d_data)
    ax.autoscale()
    plt.show()


def tsne(images, labels):

    """

    Function to return scaled tsne and its labels

    :param labels: List of labels
    :param images: List of images
    :return: Scaled tsne and labels

    """

    grayImages = []
    for image in images:
        gray = rgb2gray(image)
        gray = gray.flatten()
        grayImages.append(gray)

    pca = PCA(n_components=180)
    pca_result = pca.fit_transform(grayImages)

    tsne = TSNE(n_components=2, perplexity=30.0)
    tsne_result = tsne.fit_transform(pca_result)
    tsne_result_scaled = StandardScaler().fit_transform(tsne_result)

    float_labels = [float(label) for label in labels]

    return tsne_result_scaled, float_labels


def plot_data_graph(hist, num_epochs, model_name):

    """

    Function to save and visualize plots

    :param model_name: Name of the model
    :param hist: History of training
    :param num_epochs: Number of epochs
    :return: Plots and saves training and testing loss and accuracy

    """

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
