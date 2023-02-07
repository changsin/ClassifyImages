import os
from abc import ABC, abstractmethod

import numpy as np
from lxml import etree as ET

import src.utils
from src.constants import SW_IGNORE
from src.utils import glob_files, get_parent_folder, calculate_overlapped_area
from pathlib import Path
from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')


class Parser(ABC):
    def __init__(self):
        self.labels = []

    @abstractmethod
    def parse(self, path, file_type='*'):
        pass


class PascalVOCParser(Parser):
    def parse(self, folder, file_type='*'):
        def _parse_file(filename):
            width, height = 0, 0
            xmin, ymin, xmax, ymax = 0, 0, 0, 0

            tree = ET.parse(filename)
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


def check_dupes(boxes, threshold=0.9):
    count_dupe_labels = 0
    idx_to_remove = []

    for id1 in range(len(boxes)):
        for id2 in range(id1 + 1, len(boxes)):
            tag1, xtl1, ytl1, xbr1, ybr1 = boxes[id1]
            tag2, xtl2, ytl2, xbr2, ybr2 = boxes[id2]
            rect1 = Rectangle(float(xtl1), float(ytl1), float(xbr1), float(ybr1))
            rect2 = Rectangle(float(xtl2), float(ytl2), float(xbr2), float(ybr2))
            overlapped_area, max_area = calculate_overlapped_area(rect1, rect2)
            if overlapped_area > 0 and overlapped_area >= max_area * threshold:
                print("\tDupe {} {}".format(rect1, rect2))
                count_dupe_labels += 1
                idx_to_remove.append(id2)

    return idx_to_remove


class CVATXmlParser(Parser):
    def convert_xml(self, path_in, path_out):
        tree_out = ET.parse(path_in)
        # print("Labels are: ")

        el_root = tree_out.getroot()

        is_dupe = False

        # for each image, see if there are any overlapped labels
        #           Do not add any overlapped labels
        for image in el_root.xpath('image'):
            boxes = image.getchildren()

            tmp_boxes = []

            for box in boxes:
                label = box.attrib['label']
                xtl = float(box.attrib['xtl'])
                ytl = float(box.attrib['ytl'])
                xbr = float(box.attrib['xbr'])
                ybr = float(box.attrib['ybr'])

                tmp_boxes.append((label, xtl, ytl, xbr, ybr))

            idx_to_remove = list(set(check_dupes(tmp_boxes)))

            for id in idx_to_remove:
                box = boxes[id]
                label = box.attrib['label']
                xtl = box.attrib['xtl']
                ytl = box.attrib['ytl']
                xbr = box.attrib['xbr']
                ybr = box.attrib['ybr']

                print("Removed dupe {} {}:{}".format(image.attrib['name'], id, (label, xtl, ytl, xbr, ybr)))
                image.remove(boxes[id])

            if len(idx_to_remove) > 0:
                is_dupe = True

        if is_dupe:
            file_name_out = os.path.join(path_out, Path(path_in).stem + ".xml")
            print("***Writing to {}".format(file_name_out))

            with open(file_name_out, "wb") as xml:
                xml.write(ET.tostring(tree_out, encoding="utf-8", pretty_print=True))

    def parse(self, filename, file_type='*'):
        image_labels = []

        tree = ET.parse(filename)
        # print("Labels are: ")

        task_name = os.path.basename(get_parent_folder(filename))
        project_name = task_name
        labels = []
        # if the project file is in
        if tree.xpath('meta/task/name'):
            task_name = tree.xpath('meta/task/name')[0].text
        if tree.xpath('meta/task/project'):
            project_name = tree.xpath('meta/task/project')[0].text
        # if tree.xpath('meta/task/labels'):
        #     for label in tree.xpath('meta/task/labels'):
        #         print(label)

        for el in tree.xpath('meta/task/labels/label'):
            label = el.xpath('name')[0].text

            # # # for now, this is for dashboard labels only
            # for sub_el in el.xpath('attributes/attribute'):
            #     if sub_el.xpath('name')[0].text == 'name':
            #         values = sub_el.xpath('values')[0].text.split()
            #
            #         for value in values:
            #             value = value.strip()
            #             self.labels.append("{}@{}".format(label, value))
            #
            #     print("\"{}\", ".format(at[0].text), end="")
            self.labels.append(label)

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

                occluded = int(box.attrib['occluded'])
                z_order = int(box.attrib['z_order'])

                label = box.attrib['label']
                # wtype = box.xpath('attribute[@name="name"]')[0].text
                # daynight = box.xpath('attribute[@name="daynight"]')[0].text
                # visibility = int(box.xpath('attribute[@name="visibility"]')[0].text)
                # if box.xpath('attribute[@name="name"]'):
                #     label = "{}@{}".format(label, box.xpath('attribute[@name="name"]')[0].text)

                # add other attributes too
                # daynight = box.xpath('attribute[@name="daynight"]')[0].text
                # visibility = box.xpath('attribute[@name="visibility"]')[0].text
                # box = label, xtl, ytl, xbr, ybr, occluded, z_order, daynight, visibility
                box = label, xtl, ytl, xbr, ybr, occluded, z_order

                # # only add what we care right now
                # if label in SW_EX1:
                #     boxes.append(box)
                if label in SW_IGNORE:
                    continue

                boxes.append(box)

            image_labels.append([name, width, height, project_name, task_name, np.array(boxes)])

        return np.array(image_labels, dtype=object)


class CoCoJsonParser(Parser):
    def parse(self, filename, file_type='*'):
        data = src.utils.from_file(filename)

        task_name = os.path.basename(get_parent_folder(filename))
        project_name = task_name

        annotations = data['annotations']

        labels_dict = {}
        for label in data['categories']:
            labels_dict[label['id']] = label['name']
            self.labels.append(label['name'])
            print("\"{}\",".format(label['name']))

        boxes_dict = {}
        for annotation in annotations:
            image_id = annotation['image_id']
            bbox = annotation['bbox']
            category_id = annotation['category_id']

            boxes = []
            if image_id in boxes_dict.keys():
                boxes = boxes_dict.get(image_id)

            label_name = labels_dict[category_id]

            xbr = bbox[0] + bbox[2]
            ybr = bbox[1] + bbox[3]

            boxes.append([label_name, bbox[0], bbox[1], xbr, ybr])
            boxes_dict[image_id] = boxes

        image_labels = []
        for image in data['images']:
            image_id = image['id']
            image_name = image['file_name']
            width = image['width']
            height = image['height']

            boxes = []
            if image_id in boxes_dict.keys():
                boxes = boxes_dict.get(image_id)

            image_labels.append([image_name, width, height, project_name, task_name, np.array(boxes)])

        return np.array(image_labels, dtype=object)
