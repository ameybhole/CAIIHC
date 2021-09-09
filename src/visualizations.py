from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from skimage.color import rgb2gray
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def visualize_scatter_with_images(X_2d_data, images, figsize=(50,50), image_zoom=0.2):
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

