import datetime
import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path

from lxml import etree as ET

import src.utils
from src.constants import CHILD_ZONE


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
            for a in image_info[-1]:
                width = float(a[3]) - float(a[1])
                height = float(a[4]) - float(a[2])
                image_label = ImageLabel(a[0],
                                         (float(a[1]) + width / 2) / res_w,
                                         (float(a[2]) + height / 2) / res_h,
                                         width / res_w, height / res_h)
                labels.append(image_label)

            tokens = os.path.basename(path).split('_')
            prefix = tokens[0]
            suffix_tokens = tokens[6:-3]
            suffix = ""
            for t in suffix_tokens:
                suffix += "{}{}".format(t, "_")

            sub_folder1 = os.path.join(os.path.dirname(path), "{}_{}".format(prefix, suffix))
            sub_folder1 = sub_folder1.removesuffix("_")

            sub_folder2 = os.path.join(sub_folder1, os.path.basename(path)[:-4])
            if not os.path.exists(sub_folder2):
                os.makedirs(sub_folder2)
            # out_filename = os.path.join(sub_folder, image_filename[:-3] + 'txt')
            out_filename = os.path.join(sub_folder2, image_filename[:-3] + 'txt')

            # print(out_filename, labels)
            # print("Writing ", out_filename)
            with open(out_filename, "w+") as file_out:
                for label in labels:
                    # class_id = parser.labels.index(label.label)
                    class_id = CHILD_ZONE.index(label.label)
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
            for a in image_info[-1]:
                width = float(a[3]) - float(a[1])
                height = float(a[4]) - float(a[2])
                image_label = ImageLabel(a[0], int(float(a[1])), int(float(a[2])),
                                         int(width), int(height))
                labels.append(image_label)

            image_labels[image_filename] = labels


class CVATXmlConverter(Converter):
    def convert(self, path, parser):
        parsed = parser.parse(path)

        tree_out = ET.parse(".\\data\\labels\\cvat_dashboard.xml")
        # print("Labels are: ")
        # for el in tree_out.xpath('meta/task/labels/label'):

        datetime_now = datetime.datetime.now()
        el_created = tree_out.xpath('meta/task/created')
        el_updated = tree_out.xpath('meta/task/updated')
        el_dumped = tree_out.xpath('meta/dumped')
        el_created[0].text = str(datetime_now)
        el_updated[0].text = str(datetime_now)
        el_dumped[0].text = str(datetime_now)

        el_project_name = tree_out.xpath('meta/task/project')
        el_task_name = tree_out.xpath('meta/task/name')
        project_name = parsed[0][3]
        task_name = parsed[0][4]
        el_project_name[0].text = str(project_name)
        el_task_name[0].text = str(task_name)

        el_root = tree_out.getroot()
        if parser.labels:
            el_labels = el_root.xpath('meta/task/labels')
            children = el_labels[0].getchildren()

            for child in children:
                el_labels[0].remove(child)

            for label in parser.labels:
                el_label = ET.SubElement(el_labels[0], 'label')
                el_name = ET.SubElement(el_label, 'name')

                el_name.set('name', label)
                el_attributes = ET.SubElement(el_label, 'attributes')

        for image_info in parsed:
            image_filename = image_info[0]

            el_image = ET.SubElement(el_root, 'image')
            el_image.set('name', image_filename)
            el_image.set('width', str(image_info[1]))
            el_image.set('height', str(image_info[2]))

            el_image.set('task', str(image_info[4]))

            # print(image_info)
            #
            for a in image_info[-1]:
                el_box = ET.SubElement(el_image, 'box')

                if len(a) > 6:
                    # notice that it's the sixth token
                    el_box.set('occluded', a[5])
                    # notice that it's the sixth token
                    el_box.set('z_order', a[6])

                label = a[0]
                tokens = a[0].split('@')
                if len(tokens) > 0:
                    label = tokens[0]
                el_box.set('label', label)

                el_box.set('xtl', a[1])
                el_box.set('ytl', a[2])
                el_box.set('xbr', a[3])
                el_box.set('ybr', a[4])

                # # add attributes
                # el_name = ET.SubElement(el_box, 'attribute')
                # el_name.set('name', tokens[1])
                #
                # el_daynight = ET.SubElement(el_box, 'attribute')
                # el_daynight.set('daynight', a[7])
                #
                # el_visibility = ET.SubElement(el_box, 'attribute')
                # el_visibility.set('visibility', a[8])

        out_filename = os.path.join(os.path.dirname(path), os.path.basename(path)[:-4] + ".xml")
        with open(out_filename, "wb") as xml:
            xml.write(ET.tostring(tree_out, pretty_print=True))

    def write(self, project_name, parsed, path_out):
        tree_out = ET.parse(".\\data\\labels\\cvat_sidewalk.xml")

        datetime_now = datetime.datetime.now()
        el_created = tree_out.xpath('meta/task/created')
        el_updated = tree_out.xpath('meta/task/updated')
        el_dumped = tree_out.xpath('meta/dumped')
        el_created[0].text = str(datetime_now)
        el_updated[0].text = str(datetime_now)
        el_dumped[0].text = str(datetime_now)

        # el_project_name = tree_out.xpath('meta/task/project')
        el_task_name = tree_out.xpath('meta/task/name')
        # project_name = parsed[0][3]
        # task_name = parsed[0][4]
        # el_project_name[0].text = project_name
        el_task_name[0].text = project_name

        el_root = tree_out.getroot()
        for image_info in parsed:
            image_filename = image_info[0]

            el_image = ET.SubElement(el_root, 'image')
            el_image.set('name', image_filename)
            el_image.set('width', str(image_info[1]))
            el_image.set('height', str(image_info[2]))

            el_image.set('task', str(image_info[4]))

            # print(image_info)
            #
            for a in image_info[-1]:
                el_box = ET.SubElement(el_image, 'box')
                tokens = a[0].split('@')
                el_box.set('label', tokens[0])
                # notice that it's the sixth token
                el_box.set('occluded', a[5])

                el_box.set('xtl', a[1])
                el_box.set('ytl', a[2])
                el_box.set('xbr', a[3])
                el_box.set('ybr', a[4])

                # notice that it's the sixth token
                el_box.set('z_order', a[6])

                # # add attributes
                # el_name = ET.SubElement(el_box, 'attribute')
                # el_name.set('name', 'name')
                # el_name.text = tokens[1]

                # el_daynight = ET.SubElement(el_box, 'attribute')
                # el_daynight.set('name', 'daynight')
                # el_daynight.text = a[7]
                #
                # el_visibility = ET.SubElement(el_box, 'attribute')
                # el_visibility.set('name', 'visibility')
                # el_visibility.text = a[8]

        with open(path_out, "wb") as xml:
            xml.write(ET.tostring(tree_out, pretty_print=True))


class CoCoConverter(Converter):
    def __init__(self):
        self.categories = [
            # TODO: hard-coding it for this dataset
            {"id": 1, "name": "Animals(Dolls)"},
            {"id": 2, "name": "Person"},
            {"id": 3, "name": "Garbage bag & sacks"},
            {"id": 4, "name": "Construction signs & Parking prohibited board"},
            {"id": 5, "name": "Traffic cone"},
            {"id": 6, "name": "Box"},
            {"id": 7, "name": "Stones on road"},
            {"id": 8, "name": "Pothole on road"},
            {"id": 9, "name": "Filled pothole"},
            {"id": 10, "name": "Manhole"}
        ]

    def _find_category_id(self, name):
        matching_id = None
        for el in self.categories:
            id1, name1 = el['id'], el['name']
            # TODO: this is a hacky way of finding the id from the category name but works for this project
            if name[:3] == name1[:3]:
                matching_id = id1
                break

        assert (matching_id is not None)

        return matching_id

    @staticmethod
    def _parse_date_from_file_name(file_name):
        # V3F_HY_8484_20201208_130036_E_CH1_Seoul_Sun_Mainroad_Day_50936.png
        # date is 20201208
        matched = re.search('^[a-z0-9]*_[a-z0-9]*_[0-9]*_[0-9]*', file_name, re.IGNORECASE).group(0)
        tokens = matched.split('_')
        year_date = tokens[3]
        year, month, day = year_date[:4], year_date[4:6], year_date[-2:]
        return year, month, day

    def convert(self, path, parser):
        parsed = parser.parse(path)

        for img in parsed:
            json_labels = {}

            info = dict()
            info["description"] = img[3]
            info["url"] = ""
            info["version"] = "1.0"
            year, month, day = CoCoConverter._parse_date_from_file_name(img[0])
            info["year"] = int(year)
            info["contributor"] = "Konkuk_university"  # hard-code it for this dataset
            info["date_created"] = "{}/{}/{}".format(year, month, day)
            json_labels["info"] = info

            images = dict()
            images["file_name"] = img[0]
            images["width"] = img[1]
            images["height"] = img[2]
            images["id"] = 1
            json_labels["images"] = images

            annotations = []
            for idx, ann in enumerate(img[5]):
                annotation = dict()
                annotation["segmentation"] = []
                annotation["polyline"] = []
                annotation["image_id"] = 1

                label, xtl, ytl, xbr, ybr, _, _ = ann
                width = float(xbr) - float(xtl)
                height = float(ybr) - float(ytl)
                annotation["bbox"] = [
                    float(xtl),
                    float(ytl),
                    width,
                    height
                ]
                annotation["category_id"] = self._find_category_id(label)
                annotation["area"] = width * height
                annotation["is_crowd"] = 0  # hard-coding it for now
                annotation["id"] = idx + 1

                annotations.append(annotation)

            json_labels["annotations"] = annotations

            json_labels["categories"] = self.categories

            out_path = os.path.join(Path(path).parent, Path(img[0]).stem + ".json")
            print("Writing to {}".format(out_path))

            # Set ensure_ascii=False to write hangul and other unicode chars correctly
            src.utils.to_file(out_path, json.dumps(json_labels, ensure_ascii=False))


def default(obj):
    if hasattr(obj, 'to_json'):
        return obj.to_json()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')
