import argparse
import glob
import json
import os
from enum import Enum
from lxml import etree


import numpy as np

"""
convert label files into different formats
"""

IMAGE_SIZE = 320


class Mode(Enum):
    CONVERT = 'convert'

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


def parse_xml(filename):
    image_labels = []

    tree = etree.parse(filename)
    for image in tree.xpath('image'):
        # print(image.attrib['name'])
        name = image.attrib['name']
        width = int(image.attrib['width'])
        height = int(image.attrib['height'])

        boxes = []

        for box in image.xpath('box'):
            xtl = float(box.attrib['xtl'])
            ytl = float(box.attrib['ytl'])
            xbr = float(box.attrib['xbr'])
            ybr = float(box.attrib['ybr'])

            alertwarning = box.attrib['label']
            wtype = box.xpath('attribute[@name="name"]')[0].text
            daynight = box.xpath('attribute[@name="daynight"]')[0].text
            visibility = int(box.xpath('attribute[@name="visibility"]')[0].text)

            box = wtype, alertwarning, daynight, visibility, xtl, ytl, xbr, ybr

            boxes.append(box)

        image_labels.append([name, width, height, np.array(boxes)])

    return np.array(image_labels)


def default(obj):
    if hasattr(obj, 'to_json'):
        return obj.to_json()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


class ImageLabel:
    def __init__(self, label, x, y, width, height):
        self.label = label
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __iter__(self):
        yield from {
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height
        }.items()

    def __str__(self):
        return json.dumps(dict(self), default=default, ensure_ascii=False)

    def __repr__(self):
        return self.__str__()

    def to_json(self):
        return self.__str__()


class EdgeImpulseLabels:
    def __init__(self, bboxes):
        self.version = 1
        self.type = "bounding-box-labels"
        self.bboxes = bboxes
        if self.bboxes is None:
            self.bboxes = {}

    def __iter__(self):
        yield from {
            "version": self.version,
            "type": self.type,
            "boundingBoxes": self.bboxes
        }.items()

    def __str__(self):
        return json.dumps(dict(self), default=self.to_json, ensure_ascii=False)

    def __repr__(self):
        return json.dumps(dict(self), default=self.to_json, ensure_ascii=False)

    def to_json(self):
        to_return = {"version": self.version, "type": self.type}
        image_boxes = {}
        for key, boxes in self.bboxes.items():
            jboxes = []
            for box in boxes:
                jboxes.append(box.__dict__)
            image_boxes[key] = jboxes

        to_return["boundingBoxes"] = image_boxes
        return to_return


def xml2pascalvoc(filename):
    parsed = parse_xml(filename)

    image_labels = {}

    for image_info in parsed:
        image_filename = image_info[0]

        labels = []
        for a in image_info[3]:
            width = float(a[6]) - float(a[4])
            height = float(a[7]) - float(a[5])
            image_label = ImageLabel(a[0], int(float(a[4])), int(float(a[5])),
                                     int(width), int(height))
            labels.append(image_label)

        image_labels[image_filename] = labels
    # nums = [1, 2, 3, 4]
    # print(json.dumps(nums))

        # print(image_filename, labels[0])
        # print(image_filename, json.dumps(labels))
        # print(image_filename, labels[0].__dict__)
        # image_labels[image_filename] = labels
        # print(image_labels)

    # print(image_labels)

    eilabels = EdgeImpulseLabels(image_labels)

    # print(json.dumps(eilabels.__dict__, indent='  '))
    print(json.dumps(eilabels.to_json()))
    # print(json.dumps(eilabels, indent='  '))
    to_file(".\\" + os.path.basename(filename) + '.json', json.dumps(eilabels.to_json()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", action="store", type=Mode.argparse, choices=list(Mode), dest="mode")
    parser.add_argument("-path", action="store", dest="path", type=str)
    parser.add_argument("-path_out", action="store", dest="path_out", type=str)
    parser.add_argument("-centroids_json", action="store", dest="centroids_json", type=str)
    parser.add_argument("-threshold", action="store", dest="threshold", type=float, default=80)

    args = parser.parse_args()

    xml2pascalvoc(args.path)
