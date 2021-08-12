import cv2
import os
import glob
import numpy as np
import argparse

"""
A sample program to classify images:
1. day/night images - based on pixel values
"""

def glob_files(path):
    search_string = os.path.join(path, '*')
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

def get_avg_pixel_val(image):
  num_of_pixels = image.shape[0]*image.shape[1]
  if len(image.shape) > 2:
      num_of_pixels *= image.shape[2]

  return np.sum(image)/num_of_pixels

def detect_night_images(folder_path, threshold=80):
    paths = glob_files(folder_path)

    for path in paths:
        # image = cv2.imread(path)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            hist = get_avg_pixel_val(image)
            if hist < threshold:
                print("Detected a night image: {:.02f} {}".format(hist, path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", action="store", dest="path", type=str)
    parser.add_argument("-threshold", action="store", dest="threshold", type=int, default=80)

    args = parser.parse_args()
    detect_night_images(args.path, args.threshold)
