import argparse
import glob
import os

import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from scipy.spatial.distance import cdist
from sklearn import preprocessing  # to normalise existing X
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

"""
"""

IMAGE_SIZE = 320

def glob_files(folder, file_type='*'):
    search_string = os.path.join(folder, file_type)
    files = glob.glob(search_string)

    print('Searching ', search_string)
    paths = []
    for f in files:
      if os.path.isdir(f):
        sub_paths = glob_files(f + '/')
        paths += sub_paths
      else:
        paths.append(f)

    # We sort the images in alphabetical order to match them
    #  to the annotation files
    paths.sort()

    return paths


def get_feature_maps(path, file_type="*.png"):
    def _to_feature_maps(X):
        #Convert to VGG input format
        X = preprocess_input(X)

        #include_top=False == not getting VGG16 last 3 layers
        model = VGG16(weights="imagenet", include_top=False)
        #Get features
        features = model.predict(X)
        print(features.shape)

        return features

    files = glob_files(path, file_type)

    feature_maps = []
    for file in files:
        print(file)
        image = cv2.imread(file)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        fm = _to_feature_maps(np.array([image]))
        feature_maps.append(fm)

    return np.array(feature_maps)


def get_pca_reduced(X_features, dimensions=2):
  X_features_flatten = X_features.reshape(X_features.shape[0], -1)
  print("Flattened shape: ", X_features_flatten.shape)
  pca = PCA(dimensions)

  X_features_pca_reduced = pca.fit_transform(X_features_flatten)

  return X_features_pca_reduced, pca


def get_clusters(X_reduced, K):
  kmeans = KMeans(n_clusters=K, random_state=0)
  X_clusters = kmeans.fit(X_reduced)

  return X_clusters, kmeans

def to_cluster_idx(cluster_labels, bins):
    """
    param bins: range of K
    param labels: cluster labels
    returns: dictionary of cluster IDs
    """
    cluster_dict = dict()
    for cluster_id in bins:
        cluster_dict[cluster_id] = np.where(cluster_labels == cluster_id)[0]
    return cluster_dict

def cluster_images(folder, file_type="*.png"):
    X_fm = get_feature_maps(folder, file_type=file_type)
    print("####", X_fm.shape)

    # normalize
    X_fm_normalized = preprocessing.normalize(X_fm.reshape(len(X_fm), -1))

    print(X_fm_normalized.shape)
    # # dimensionality reduction through PCA
    X_reduced, pca = get_pca_reduced(X_fm_normalized, dimensions=2)

    # # cluster
    X_clusters, kmeans = get_clusters(X_reduced, 2)
    # X_clusters, kmeans = get_clusters(X_fm_normalized, 2)

    # get the image ids of each cluster
    cluster_idx = to_cluster_idx(X_clusters.labels_, [0, 1])

    # keep the cluster centers
    print(kmeans.cluster_centers_)
    print(cluster_idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", action="store", dest="path", type=str)
    parser.add_argument("-threshold", action="store", dest="threshold", type=int, default=80)

    args = parser.parse_args()
    cluster_images(args.path)
