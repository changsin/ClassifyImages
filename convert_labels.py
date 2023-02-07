import argparse
import os
import shutil

from src.constants import Mode, LabelFormat
from src.converter import YoloV5Converter, EdgeImpulseConverter, CVATXmlConverter, CoCoConverter
from src.parser import CoCoJsonParser, CVATXmlParser, PascalVOCParser
from src.utils import glob_files, glob_folders, split_train_val_test_files, copy_label_files, flat_copy, calculate_overlapped_area
from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

"""
convert label files into different formats
"""


def convert_labels(path, from_format, to_format=LabelFormat.EDGE_IMPULSE):
    parser = None
    convertor = None

    if from_format == LabelFormat.CVAT_XML:
        parser = CVATXmlParser()
    elif from_format == LabelFormat.PASCAL_VOC:
        parser = PascalVOCParser()
    elif from_format == LabelFormat.COCO_JSON:
        parser = CoCoJsonParser()
    else:
        print('Unsupported input format {}'.format(from_format))

    if to_format == LabelFormat.EDGE_IMPULSE:
        convertor = EdgeImpulseConverter()
    elif to_format == LabelFormat.YOLOV5:
        convertor = YoloV5Converter()
    elif to_format == LabelFormat.CVAT_XML:
        convertor = CVATXmlConverter()
    elif to_format == LabelFormat.COCO_JSON:
        convertor = CoCoConverter()
    else:
        print('Unsupported output format {}'.format(to_format))

    convertor.convert(path, parser)


def check_overlaps(path_in, path_out, from_format):
    parser = None

    if from_format == LabelFormat.CVAT_XML:
        parser = CVATXmlParser()
    else:
        print('Unsupported input format {}'.format(from_format))

    parsed = parser.parse(path_in)

    overlap_dict = dict()
    overlap_dict[0.95] = 0
    overlap_dict[0.90] = 0
    overlap_dict[0.85] = 0
    overlap_dict[0.80] = 0
    overlap_dict[0.75] = 0

    count_dupe_labels = 0
    for image in parsed:
        labels = image[5]
        for id1 in range(len(labels)):
            for id2 in range(id1 + 1, len(labels)):
                tag1, xtl1, ytl1, xbr1, ybr1, _, _ = labels[id1]
                tag2, xtl2, ytl2, xbr2, ybr2, _, _ = labels[id2]
                rect1 = Rectangle(float(xtl1), float(ytl1), float(xbr1), float(ybr1))
                rect2 = Rectangle(float(xtl2), float(ytl2), float(xbr2), float(ybr2))
                overlapped_area, max_area = calculate_overlapped_area(rect1, rect2)
                #
                #         if intersect_area >= max_area*threshold:
                # #           if intersect_area >= min_area * threshold and intersect_area < min_area * 0.95:
                #             overlapped_area = intersect_area
                if overlapped_area > 0:
                    count_dupe_labels += 1
                    for threshold, count in overlap_dict.items():
                        if overlapped_area >= max_area * threshold:
                            overlap_dict[threshold] = overlap_dict[threshold] + 1

    if count_dupe_labels > 0:
        print("Dupes {}: {} {}".format(path_in, count_dupe_labels, overlap_dict))

    return overlap_dict


def convert_xmls(path_in, path_out):
    parser = CVATXmlParser()

    parser.convert_xml(path_in, path_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", action="store", type=Mode.argparse, choices=list(Mode), dest="mode")
    parser.add_argument("--format_in", action="store", type=LabelFormat.argparse, choices=list(LabelFormat), dest="format_in")
    parser.add_argument("--format_out", action="store", type=LabelFormat.argparse, choices=list(LabelFormat), dest="format_out")
    parser.add_argument("--path_in", action="store", dest="path_in", type=str)
    parser.add_argument("--path_out", action="store", dest="path_out", type=str)

    parser.add_argument("--path_images", action="store", dest="path_images", type=str)
    parser.add_argument("--path_labels", action="store", dest="path_labels", type=str)

    args = parser.parse_args()
    print(args.mode)

    if args.mode == Mode.REMOVE_UNLABELED_FILES:
        parent_folder = args.path_in[:args.path_in[:-2].rfind('\\'):]
        args.path_out = os.path.join(parent_folder, "unlabeled")
        if not os.path.exists(args.path_out):
            os.mkdir(args.path_out)

        if os.path.isdir(args.path_in):
            # files = glob_files(args.path, file_type='*')
            files = glob_files(args.path_in, file_type='*.jpg')

            for file in files:
                txt_file = os.path.basename(file)[:-3] + 'txt'
                txt_file = os.path.join(os.path.dirname(file), txt_file)
                if not os.path.exists(txt_file):
                    print('does not have a label file:', txt_file)
                    dest = os.path.join(args.path_out, os.path.basename(file))
                    shutil.move(file, dest)

    elif args.mode == Mode.CONVERT_XML:

        if os.path.isdir(args.path_in):
            files = []
            folders = glob_folders(args.path_in, file_type='*')
            if folders:
                for folder in folders:
                    files.extend(glob_files(folder, file_type='*.xml'))
            # print(files)
            else:
                files = glob_files(args.path_in, file_type='*.xml')

            print(files)

            for file in files:
                convert_xmls(file, args.path_out)
        else:
            convert_xmls(args.path_in, args.path_out)

    elif args.mode == Mode.CONVERT:

        if os.path.isdir(args.path_in):
            files = []
            folders = glob_folders(args.path_in, file_type='*')
            if folders:
                for folder in folders:
                    files.extend(glob_files(folder, file_type='*.xml'))
            # print(files)
            else:
                files = glob_files(args.path_in, file_type='*.xml')

            print(files)

            for file in files:
                convert_labels(file, args.format_in, args.format_out)
        else:
            convert_labels(args.path_in, args.format_in, args.format_out)

    elif args.mode == Mode.SPLIT:
        split_train_val_test_files(args.path_in, args.path_in, args.path_out, ratio=0.1)

    elif args.mode == Mode.COPY_LABEL_FILES:
        copy_label_files(args.path_images, args.path_labels)

    elif args.mode == Mode.FLAT_COPY:
        flat_copy(args.path_in, args.path_out)

    elif args.mode == Mode.CHECK_OVERLAPS:
        if os.path.isdir(args.path_in):
            files = []
            folders = glob_folders(args.path_in, file_type='*')
            if folders:
                for folder in folders:
                    files.extend(glob_files(folder, file_type='*.xml'))
            # print(files)
            else:
                files = glob_files(args.path_in, file_type='*.xml')

            print(files)

            overlap_dict = dict()
            overlap_dict[0.95] = 0
            overlap_dict[0.90] = 0
            overlap_dict[0.85] = 0
            overlap_dict[0.80] = 0
            overlap_dict[0.75] = 0

            for file in files:
                dict1 = check_overlaps(file, args.path_out, args.format_in)
                for threshold, count in dict1.items():
                    overlap_dict[threshold] = overlap_dict[threshold] + count

            print("Total dupes: {}".format(overlap_dict))
        else:
            check_overlaps(args.path_in, args.path_out, args.format_in)

    else:
        print("Please specify the mode")
