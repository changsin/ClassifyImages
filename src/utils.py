import glob
import json
import os
import random
import shutil
from pathlib import Path

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


def glob_folders(folder, file_type='*'):
    search_string = os.path.join(folder, file_type)
    files = glob.glob(search_string)

    print('Searching ', search_string)
    paths = []
    for f in files:
      if os.path.isdir(f):
        paths.append(f)

    # We sort the images in alphabetical order to match them
    #  to the annotation files
    paths.sort()

    return paths


def to_file(path, data):
    """
    save data to path
    """
    with open(path,  'w', encoding="utf-8") as json_file:
        json_file.write(data)


def from_file(path):
    """
    save json data to path
    """
    file = open(path, 'r', encoding='utf-8')
    return json.load(file)


def get_parent_folder(path):
    return path[:path[:-2].rfind('\\'):]


def split_train_val_test_files(parent_folder, folder_from, folder_to, ratio=0.1):
    def _copy_files(files_from, folder_to, start_id, end_id):
        for id in range(start_id, end_id):
            file_from = files_from[id]

            sub_folder_parent = os.path.basename(os.path.dirname(file_from))
            sub_folder_to = os.path.join(folder_to, os.path.basename(sub_folder_parent))
            file_to = os.path.join(sub_folder_to, os.path.basename(file_from))
            if not os.path.exists(sub_folder_to):
                print("Creating folder to ", sub_folder_to)
                os.mkdir(sub_folder_to)

            if os.path.exists(file_to):
                print("ERROR: target {} already exists".format(file_to))
                print("Skipping")
                continue
                # exit(-1)

            else:
                print(file_from, file_to)
                shutil.copy(file_from, file_to)

    folder_to = os.path.join(parent_folder, folder_to)
    folder_train = os.path.join(folder_to, "train")
    folder_val = os.path.join(folder_to, "val")
    folder_test = os.path.join(folder_to, "test")

    if not os.path.exists(folder_to):
        print("Creating folder to ", folder_to)
        os.mkdir(folder_to)
    if not os.path.exists(folder_train):
        print("Creating folder to ", folder_train)
        os.mkdir(folder_train)
    if not os.path.exists(folder_val):
        print("Creating folder to ", folder_val)
        os.mkdir(folder_val)
    if not os.path.exists(folder_test):
        print("Creating folder to ", folder_test)
        os.mkdir(folder_test)

    sub_folders = glob_folders(folder_from)
    copied_count = 0

    for sub_id, sub_folder in enumerate(sub_folders):
        files = glob_files(sub_folder)

        random.shuffle(files)
        end_id = len(files)
        test_id = int(len(files) * 0.1)
        print("Copying {} - {} files".format(0, test_id))
        val_id = test_id * 2
        print("Copying {} - {} files".format(test_id, val_id))

        _copy_files(files, folder_test, 0, test_id)
        _copy_files(files, folder_val, test_id, val_id)
        _copy_files(files, folder_train, val_id, end_id)
        copied_count += end_id

    print("Copied ", copied_count)


def copy_label_files(folder_images, folder_labels):
    count_copied = 0
    count_skipped = 0
    count_images = 0
    count_labels = 0
    count_dupe_img = 0

    labels_dict = dict()
    label_sub_folders = glob_folders(folder_labels)
    for label_sub_id, label_sub_folder in enumerate(label_sub_folders):
        label_files = glob_files(label_sub_folder)

        count_labels += len(label_files)

        for label_file in label_files:
            file_name = Path(os.path.basename(label_file)).stem
            if labels_dict.get(file_name.lower()):
                print("***ERROR: Duplicate label file names found! {} {}"
                .format(label_file, labels_dict[file_name.lower()]))
            else:
                labels_dict[file_name.lower()] = label_file

    image_sub_folders = glob_folders(folder_images)
    images_dict = dict()
    for img_sub_id, img_sub_folder in enumerate(image_sub_folders):
        img_files = glob_files(img_sub_folder, file_type='*.jpg')

        count_images += len(img_files)

        for img_file in img_files:
            file_name = Path(os.path.basename(img_file)).stem
            if images_dict.get(file_name.lower()):
                print("***ERROR: Duplicate image file names found! {} {}"
                .format(img_file, images_dict[file_name.lower()]))
                count_dupe_img += 1
            else:
                images_dict[file_name.lower()] = img_file

            if not labels_dict.get(file_name.lower()):
                print("*Error: can't find the matching label file for {}".format(file_name))
                count_skipped += 1
            else:
                label_file_path = labels_dict[file_name.lower()]
                print(label_file_path, img_sub_folder)
                shutil.copy(label_file_path, os.path.dirname(img_file))
                count_copied += 1

    print("Found {} duplicate image files".format(count_dupe_img))
    print("Copied {} label files: {}/{} skipped: {}".format(count_copied, count_labels, count_images, count_skipped))


# Copy and flatten files - meaning that all files will be copied as a single dir under path_out
def flat_copy(path_in, path_out):
    copied_count = 0

    sub_folders = glob_folders(path_in)
    for sub_id, sub_folder in enumerate(sub_folders):
        files = glob_files(sub_folder)

        print("Copied {} files in {}".format(len(files), sub_folder))
        for file in files:
            file_to = os.path.join(path_out, os.path.basename(file))
            shutil.copy(file, file_to)
            copied_count += 1
    print("Copied {} files".format(copied_count))


def calculate_overlapped_area(a, b):
    overlapped_area = 0.0
    max_area = 0.0

    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx >= 0) and (dy >= 0):
        area1 = (a.xmax - a.xmin) * (a.ymax - a.ymin)
        area2 = (b.xmax - b.xmin) * (b.ymax - b.ymin)

        max_area = max(area1, area2)
        overlapped_area = dx*dy

    return overlapped_area, max_area
