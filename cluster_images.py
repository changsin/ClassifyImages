import argparse
import glob
import json
import os
from enum import Enum

import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from scipy.spatial.distance import cdist
from sklearn import preprocessing  # to normalise
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

"""
Cluster images using CNN feature maps and PCA.
"""

IMAGE_SIZE = 320


class Mode(Enum):
    CLUSTER = 'cluster'
    CLASSIFY = 'classify'

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return Mode[s.upper()]
        except KeyError:
            return s


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


def to_feature_maps(path, file_type="*.png"):
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

    files_processed = []
    feature_maps = []
    for file in files:
        print(file)
        image = cv2.imread(file)
        if image is not None:
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            # doing it one at a time to reduce the memory foot print
            fm = _to_feature_maps(np.array([image]))
            feature_maps.append(fm)
            files_processed.append(file)
        else:
            print(file, ' is not an image file')

    return np.array(feature_maps), files_processed


def to_pca_reduced(x_features, dimensions=2):
    """
    reduces dimensions of the input.
    This is mainly for plotting purposes.
    :param x_features: feature maps
    :param dimensions: target number of dimensions
    :return: reduced features
    """
    X_features_flatten = x_features.reshape(x_features.shape[0], -1)
    print("Flattened shape: ", X_features_flatten.shape)
    pca = PCA(dimensions)

    X_features_pca_reduced = pca.fit_transform(X_features_flatten)

    return X_features_pca_reduced, pca


def to_clusters(x_reduced, K):
  kmeans = KMeans(n_clusters=K, random_state=0)
  X_clusters = kmeans.fit(x_reduced)

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

def cluster_images(folder, file_type="*"):
    X_fm, filenames = to_feature_maps(folder, file_type=file_type)
    print("####", X_fm.shape)

    # normalize to use cosine similarity
    X_fm_normalized = preprocessing.normalize(X_fm.reshape(len(X_fm), -1))

    print(X_fm_normalized.shape)

    # number of clusters
    K = 2

    # # Dimensionality reduction through PCA.
    # # This is optional.
    # # We are using it mainly for plotting purposes.
    # X_reduced, pca = to_pca_reduced(X_fm_normalized, dimensions=K)

    # cluster using feature maps or PCA reduced features
    # X_clusters, kmeans = to_clusters(X_reduced, K)
    X_clusters, kmeans = to_clusters(X_fm_normalized, K)

    # get the image ids of each cluster
    cluster_idx = to_cluster_idx(X_clusters.labels_, range(K))

    # keep the cluster centers
    print(kmeans.cluster_centers_)
    print(cluster_idx)
    centroids_filename = 'centroids.json'
    to_json(centroids_filename, kmeans.cluster_centers_.tolist())
    print('centroid values are saved as {}'.format(centroids_filename))

    for key, idx in cluster_idx.items():
        print("Cluster {}".format(key))

        for id in idx:
            print("\t{}".format(filenames[id]))

def to_json(path, data):
    """
    save json data to path
    """
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def from_json(path):
    """
    save json data to path
    """
    file = open(path, 'r', encoding='utf-8')
    return json.load(file)

def classify(folder, centroids_file, file_type='*', threshold=0.7):
    centroids = from_json(centroids_file)
    X_fm, filenames = to_feature_maps(folder, file_type=file_type)
    print("####", X_fm.shape)

    # normalize to use cosine similarity
    X_fm = preprocessing.normalize(X_fm.reshape(len(X_fm), -1))

    # use cosine to calculate similarities
    dist = cdist(X_fm, centroids, metric='cosine')
    print(dist)

    for id, d, filename in zip(range(len(filenames)), dist, filenames):
        cluster_id = np.argmin(d)
        dist_min = np.min(d)

        print("{}: {} is {}".format(id, os.path.basename(filename), cluster_id))
        if dist_min > threshold:
            print("\t{} might not belong to any cluster. {}".format(os.path.basename(filename), d))
            print("\tTime to create a new cluster")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", action="store", type=Mode.argparse, choices=list(Mode), dest="mode")
    parser.add_argument("-path", action="store", dest="path", type=str)
    parser.add_argument("-centroids_json", action="store", dest="centroids_json", type=str)
    parser.add_argument("-threshold", action="store", dest="threshold", type=int, default=80)

    args = parser.parse_args()

    if args.mode == Mode.CLUSTER:
        cluster_images(args.path)
    elif args.mode == Mode.CLASSIFY:
        classify(args.path, args.centroids_json)
    else:
        raise ValueError("Specify either -cluster or -classify option")
