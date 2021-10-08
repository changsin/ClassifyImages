import argparse
import glob
import json
import os
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from lxml import etree

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


class LabelFormat(Enum):
    CVAT_XML        = 'cvat_xml'
    KAGGLE_XML      = 'kaggle_xml'
    EDGE_IMPULSE    = 'edge_impulse'
    YOLOV5          = 'yolov5'

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return LabelFormat[s.upper()]
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


class Parser(ABC):
    def __init__(self):
        self.labels = []

    @abstractmethod
    def parse(self, path, file_type='*'):
        pass


class KaggleXmlParser(Parser):
    def parse(self, folder, file_type='*'):
        def _parse_file(filename):
            width, height = 0, 0
            xmin, ymin, xmax, ymax = 0, 0, 0, 0

            tree = etree.parse(filename)
            for dim in tree.xpath("size"):
                width = int(dim.xpath("width")[0].text)
                height = int(dim.xpath("height")[0].text)

            for dim in tree.xpath("object/bndbox"):
                xmin = int(dim.xpath("xmin")[0].text)
                ymin = int(dim.xpath("ymin")[0].text)
                xmax = int(dim.xpath("xmax")[0].text)
                ymax = int(dim.xpath("ymax")[0].text)

            return [width, height,
                    [["license", xmin, ymin, xmax, ymax]]]

        files = glob_files(folder, file_type=file_type)

        image_labels = []

        for file in files:
            width, height, image_label = _parse_file(file)
            basename = os.path.basename(file).replace('.xml', '.png')

            image_labels.append([basename, width, height, np.array(image_label)])

        return np.array(image_labels)


class CVATXmlParser(Parser):
    def parse(self, filename, file_type='*'):
        image_labels = []

        tree = etree.parse(filename)
        print("Labels are: ")
        for el in tree.xpath('meta/task/labels/label'):
            label = el.xpath('name')[0].text

            # # for now, this is for dashboard labels only
            for sub_el in el.xpath('attributes/attribute'):
                if sub_el.xpath('name')[0].text == 'name':
                    values = sub_el.xpath('values')[0].text.split()

                    for value in values:
                        value = value.strip()
                        self.labels.append("{}@{}".format(label, value))
            #
            #     print("\"{}\", ".format(at[0].text), end="")
            #     self.labels.append(at)

        self.labels.sort()

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

                label = box.attrib['label']
                # wtype = box.xpath('attribute[@name="name"]')[0].text
                # daynight = box.xpath('attribute[@name="daynight"]')[0].text
                # visibility = int(box.xpath('attribute[@name="visibility"]')[0].text)
                if box.xpath('attribute[@name="name"]'):
                    label = "{}@{}".format(label, box.xpath('attribute[@name="name"]')[0].text)

                box = label, xtl, ytl, xbr, ybr

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


class Converter(ABC):
    @abstractmethod
    def convert(self, path, parser):
        pass

# dashboard classes
DASHBOARD_CLASSES = [
                    "alert@Alternator", "alert@Brake", "alert@Coolant",
                    "alert@Distance", "alert@EngineOil", "alert@EngineOilTemp",
                    "alert@Parking", "alert@Retaining", "alert@Seatbelt",
                    "alert@Steering",

                    "warning@ABS", "warning@Brake", "warning@BrakeWear",
                    "warning@CentralMonitoring", "warning@EPC", "warning@Engine",
                    "warning@Fuel", "warning@Glow", "warning@Headlamp",
                    "warning@Lamp", "warning@Parking", "warning@Retaining",
                    "warning@StabilityOff", "warning@StabilityOn", "warning@Steering",
                    "warning@TPMS", "warning@Tire", "warning@Washer"
]

SIDEWALK_CLASSES = [
    "wheelchair", "truck", "tree_trunk", "traffic_sign", "traffic_light",
    "traffic_light_controller", "table", "stroller", "stop", "scooter",
    "potted_plant", "power_controller", "pole", "person", "parking_meter",
    "movable_signage", "motorcycle", "kiosk", "fire_hydrant", "dog",
    "chair", "cat", "carrier", "car", "bus",
    "bollard", "bicycle", "bench", "barricade" ]


class YoloV5Converter(Converter):
    def convert(self, path, parser):
        """
        # change to yolo v5 format
        # https://github.com/ultralytics/yolov5/issues/12
        # [x_top_left, y_top_left, x_bottom_right, y_bottom_right] to
        # [x_center, y_center, width, height]
        """

        parsed = parser.parse(path)

        for image_info in parsed:
            image_filename = image_info[0]
            res_w = image_info[1]
            res_h = image_info[2]

            labels = []
            for a in image_info[3]:
                width = float(a[3]) - float(a[1])
                height = float(a[4]) - float(a[2])
                image_label = ImageLabel(a[0],
                                         (float(a[1]) + width/2)/res_w,
                                         (float(a[2]) + height/2)/res_h,
                                         width/res_w, height/res_h)
                labels.append(image_label)

            sub_folder = os.path.join(os.path.dirname(path), os.path.basename(path)[:-4])
            out_filename = os.path.join(sub_folder, image_filename[:-3] + 'txt')
            # out_filename = os.path.join(os.path.dirname(path), image_filename[:-3] + 'txt')

            # print(out_filename, labels)
            print("Writing ", out_filename)
            with open(out_filename, "w+") as file_out:
                for label in labels:
                    # class_id = parser.labels.index(label.label)
                    class_id = DASHBOARD_CLASSES.index(label.label)
                    file_out.write("{} {} {} {} {}\n".format(class_id,
                                                             label.x, label.y, label.width, label.height))
        # [print(label) for label in enumerate(parser.labels)]
        [print("\"{}\",".format(label)) for label in parser.labels]


class EdgeImpulseConverter(Converter):
    def convert(self, path, parser):
        parsed = parser.parse(path)

        image_labels = {}

        for image_info in parsed:
            image_filename = image_info[0]

            # print(image_info)

            labels = []
            for a in image_info[3]:
                width = float(a[3]) - float(a[1])
                height = float(a[4]) - float(a[2])
                image_label = ImageLabel(a[0], int(float(a[1])), int(float(a[2])),
                                         int(width), int(height))
                labels.append(image_label)

            image_labels[image_filename] = labels


class CVATXmlConverter(Converter):
    def convert(self, path, parser):
        parsed = parser.parse(path)

        image_labels = {}

        for image_info in parsed:
            image_filename = image_info[0]

            # print(image_info)

            labels = []
            for a in image_info[3]:
                width = float(a[3]) - float(a[1])
                height = float(a[4]) - float(a[2])
                image_label = ImageLabel(a[0], int(float(a[1])), int(float(a[2])),
                                         int(width), int(height))
                labels.append(image_label)

            image_labels[image_filename] = labels


def convert_labels(path, from_format, to_format=LabelFormat.EDGE_IMPULSE):
    parser = None
    convertor = None

    if from_format == LabelFormat.CVAT_XML:
        parser = CVATXmlParser()
    elif from_format == LabelFormat.KAGGLE_XML:
        parser = KaggleXmlParser()
    else:
        print('Unsupported input format {}'.format(from_format))

    if to_format == LabelFormat.EDGE_IMPULSE:
        convertor = EdgeImpulseConverter()
    elif to_format == LabelFormat.YOLOV5:
        convertor = YoloV5Converter()
    elif to_format == LabelFormat.CVAT_XML:
        convertor = CVATXmlConverter()
    else:
        print('Unsupported output format {}'.format(to_format))

    convertor.convert(path, parser)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", action="store", type=Mode.argparse, choices=list(Mode), dest="mode")
    parser.add_argument("-format_in", action="store", type=LabelFormat.argparse, choices=list(LabelFormat), dest="format_in")
    parser.add_argument("-format_out", action="store", type=LabelFormat.argparse, choices=list(LabelFormat), dest="format_out")
    parser.add_argument("-path", action="store", dest="path", type=str)
    parser.add_argument("-path_out", action="store", dest="path_out", type=str)

    args = parser.parse_args()

    if os.path.isdir(args.path):
        files = glob_files(args.path, file_type='*.xml')

        for file in files:
            print(file)
            convert_labels(file, args.format_in, args.format_out)
    else:
        convert_labels(args.path, args.format_in, args.format_out)
