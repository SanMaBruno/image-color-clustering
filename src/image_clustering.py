import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from skimage.io import imread, imsave
from skimage.transform import resize
import matplotlib.pyplot as plt

# Función para cargar y redimensionar la imagen
def load_and_resize_image(image_path, size=(200, 200)):
    image = imread(image_path)
    image_resized = resize(image, size)
    pixels = image_resized.reshape(-1, 3)
    return pixels, image_resized

# Función para aplicar KMeans
def apply_kmeans(pixels, n_clusters=6):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pixels)
    palette = np.clip(kmeans.cluster_centers_, 0, 1)
    return labels, palette

# Función para aplicar Clustering Jerárquico
def apply_hierarchical(pixels, n_clusters=6):
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hierarchical.fit_predict(pixels)
    return labels

# Función para aplicar Gaussian Mixtures
def apply_gmm(pixels, n_components=6):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    labels = gmm.fit_predict(pixels)
    return labels

# Función para recomponer la imagen
def reconstruct_image(labels, palette, image_shape):
    reconstructed_image = palette[labels].reshape(image_shape)
    reconstructed_image_uint8 = (reconstructed_image * 255).astype(np.uint8)
    return reconstructed_image_uint8

# Función para mostrar y guardar la imagen
def show_and_save_image(image, title, output_path):
    plt.imshow(image)
    plt.axis('off')
    plt.title(title)
    plt.show()
    imsave(output_path, image)

