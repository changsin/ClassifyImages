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

def glob_files(folder, file_type='*'):
    search_string = os.path.join(folder, file_type)
    files = glob.glob(search_string)

    # print('searching ', path)
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

def load_images(path, file_type="*"):
  files = glob_files(path, file_type)

  # print(files)
  X_data = []
  for file in files:
    image = cv2.imread(file)
    # reduce image sizes
    x = cv2.resize(image, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    x = np.expand_dims(x, axis=0)
    if len(X_data) > 0:
        X_data = np.concatenate((X_data, x))
    else:
        X_data = x

    # X_data.append(image)
  X_data = np.array(X_data)
  print("Loaded {}".format(X_data.shape))
  return X_data

def find_duplicates(X_train_pca, threshold=0.1):
    # Calculate distances of all points
    distances = cdist(X_train_pca, X_train_pca)

    # Find duplicates (very similar images)
    # dupes = np.array([np.where(distances[id] < 1) for id in range(distances.shape[0])]).reshape(-1)
    dupes = [np.array(np.where(distances[id] < threshold)).reshape(-1).tolist() \
            for id in range(distances.shape[0])]

    to_remove = set()
    for d in dupes:
        if len(d) > 1:
            for id in range(1, len(d)):
                to_remove.add(d[id])
    logger.info("Found {} duplicates".format(len(to_remove)))
    return to_remove

#Calculate similar matrics
def cosine_similarity(ratings):
    sim = ratings.dot(ratings.T)
    if not isinstance(sim,np.ndarray):
        sim = sim.toarray()
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim/norms/norms.T)


def get_feature_maps(X):
    #Convert to VGG input format
    X = preprocess_input(X)

    #include_top=False == not getting VGG16 last 3 layers
    model = VGG16(weights="imagenet", include_top=False)
    # model = VGG16(weights = None, include_top=False)
    #Get features
    features = model.predict(X)
    print(features.shape)

    return features


def get_pca_reduced(X_features, dimensions=2):
  X_features_flatten = X_features.reshape(X_features.shape[0], -1)
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

def cluster_images(folder):
    X_data = load_images(folder, "*.png")

    # get feature maps
    X_features = get_feature_maps(X_data)

    # normalize
    X_fm_normalized = preprocessing.normalize(X_features.reshape(len(X_features), -1))

    # dimensionality reduction through PCA
    X_reduced, pca = get_pca_reduced(X_fm_normalized, dimensions=2)

    # cluster
    X_clusters, kmeans = get_clusters(X_reduced, 2)

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
