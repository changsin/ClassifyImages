import argparse
import os
import random
import shutil

from convert_labels import EdgeImpulseConverter, YoloV5Converter, CVATXmlConverter
from src.constants import Mode
from src.constants import SW_TOP15, LabelFormat, SW_IGNORE
from src.parser import CVATXmlParser, KaggleXmlParser
from src.utils import glob_files, glob_folders


def filter_by_labels(parsed, labels, is_in=True):
    # check for file name dupes
    def _is_dupe(image_info, flist):
        for f in flist:
            if f[0] == image_info[0]:
                print("Dupe found {}".format(f))
                return True
        return False

    filtered = []
    dupe_count = 0
    for image_info in parsed:
        if _is_dupe(image_info, filtered):
            dupe_count += 1
            # print("Skipping")
            continue

        match_found = False
        for box in image_info[-1]:
            if box[0] in labels:
                match_found = True
                # print("Adding ", image_info[0], label, box)
                if is_in:
                    filtered.append(image_info)
                    break

        # if it's a negative case, wait till all labels are tried before adding to filtered
        if not is_in and not match_found:
            filtered.append(image_info)

    return filtered, dupe_count


def filter_by_visibility(parsed, visibilities, is_in=True):
    # check for file name dupes
    def _is_dupe(image_info, flist):
        for f in flist:
            if f[0] == image_info[0]:
                # print("Dupe found {}".format(f))
                return True
        return False

    filtered = []
    dupe_count = 0
    for image_info in parsed:
        if _is_dupe(image_info, filtered):
            dupe_count += 1
            # print("Skipping")
            continue

        match_found = False
        for box in image_info[-1]:
            if box[-1] in visibilities:
                match_found = True
                # print("Adding ", image_info[0], label, box)
                if is_in:
                    filtered.append(image_info)
                    break

        # if it's a negative case, wait till all labels are tried before adding to filtered
        if not is_in and not match_found:
            filtered.append(image_info)

    return filtered, dupe_count


def count_labels(image_infos):
    label_counts = dict()

    for im in image_infos:
        label_infos = im[-1]
        for label_info in label_infos:
            label = label_info[0]
            if label in label_counts:
                label_counts[label] = label_counts[label] + 1
            else:
                label_counts[label] = 1

    label_counts = dict(sorted(label_counts.items(), key=lambda x: x[1], reverse=True))
    return label_counts


def combine_dicts(dict1, dict2):
    dict3 = dict()

    for key1, val1 in dict1.items():
        dict3[key1] = val1
        if key1 in dict2.keys():
            dict3[key1] = val1 + dict2[key1]

    # Add items not in dict2
    for key2, val2 in dict2.items():
        if key2 not in dict1.keys():
            dict3[key2] = val2

    return dict3


def pick_files(filtered, picked_filenames, label_counts, label, count=100):
    picked = []

    for item in filtered:
        if label in label_counts.keys() and label_counts[label] > count:
            break

        filename = item[0]
        if filename in picked_filenames:
            continue

        picked.append(item)
        cur_label_counts = count_labels([item])
        label_counts = combine_dicts(label_counts, cur_label_counts)

    return picked, label_counts


def get_min_key(label_counts):
    min_val = 9999
    min_key = None

    for key, value in label_counts.items():
        if min_val >= value:
            min_val = value
            min_key = key
            # break immediately, if min_val is 0
            # since this is the absolute minimum possible value
            if min_val == 0:
                break

    return min_key


def filter_balance(parsed, to_pick_labels, max_count=1000, is_in=True):
    picked_filenames = set()
    label_counts = {}
    for label in SW_TOP15:
        label_counts[label] = 0

    print('parsed labels: ', count_labels(parsed))
    picked = []
    for to_pick_label in to_pick_labels:
        pick_from, _ = filter_by_labels(parsed, [to_pick_label], is_in)
        if len(pick_from) > 0:
            (picked_loc, label_counts) = pick_files(pick_from, picked_filenames, label_counts, to_pick_label, max_count)
            # print(to_pick_label, label_counts, len(picked_loc))

            for a in picked_loc:
                filename = a[0]
                picked_filenames.add(filename)
                picked.append(a)

    print("Picked: ", label_counts, len(picked))
    return picked, None


def filter_balance1(parsed, to_pick_labels, max_count=1000, is_in=True):

    picked_filenames = set()
    label_counts = {}
    for label in SW_TOP15:
        label_counts[label] = 0

    print('parsed labels: ', count_labels(parsed))
    picked = []
    while len(picked_filenames) < max_count:
        to_pick_label = get_min_key(label_counts)
        pick_from, _ = filter_by_labels(parsed, [to_pick_label], is_in)
        if len(pick_from) > 0:
            (picked_loc, label_counts) = pick_files(pick_from, picked_filenames, label_counts, to_pick_label, max_count)
            # print(to_pick_label, label_counts, len(picked_loc))

            for a in picked_loc:
                filename = a[0]
                picked_filenames.add(filename)
                picked.append(a)

    print("Picked: ", label_counts, len(picked))
    return picked, None


def filter_files(path_in, from_format, to_format=LabelFormat.EDGE_IMPULSE):
    cvat_files = glob_files(args.path_in, file_type='*.xml')
    if cvat_files is None or len(cvat_files) == 0:
        folders = glob_folders(args.path_in, file_type='*')
        for folder in folders:
            cvat_files.extend(glob_files(folder, file_type='*.xml'))
    print(cvat_files)

    parser = None
    convertor = None

    if from_format == LabelFormat.CVAT_XML:
        parser = CVATXmlParser()
    elif from_format == LabelFormat.KAGGLE_XML:
        parser = KaggleXmlParser()
    else:
        print('Unsupported input format {}'.format(from_format))

    parsed = []
    for file in cvat_files:
        p = parser.parse(file)
        parsed.extend(p)

    # filtered = parsed
    # labels_to_filter = SW_EX1
    # filtered, dupe_count = filter_by_labels(parsed, SW_EXCLUDE, is_in=False)
    # filtered, dupe_count = filter_by_labels(filtered, TOP15, is_in=False)
    # filtered, dupe_count = filter_by_labels(filtered, TOP10, is_in=False)
    # filtered, dupe_count = filter_by_labels(filtered, ["alert@Seatbelt"])
    # filtered, dupe_count = filter_by_labels(parsed, SW_EX1)
    # filtered, dupe_count = filter_by_visibility(filtered, ['1', '2'])
    # filtered, dupe_count = filter_by_labels(parsed, SW_IGNORE, is_in=False)
    filtered, dupe_count = filter_by_labels(parsed, SW_IGNORE, is_in=False)
    filtered, dupe_count = filter_balance(parsed, SW_TOP15, max_count=1000)
    # filtered, dupe_count = filter_balance(parsed, SW_TOP15, max_count=500)
    # filtered = parsed
    dupe_count = 0
    print(len(filtered), "dupe:", dupe_count)
    for id, item in enumerate(count_labels(filtered).items()):
        print(f'{id}\t{item[0]}\t{item[1]}')
    print(count_labels(filtered))
    exit(0)

    if to_format == LabelFormat.EDGE_IMPULSE:
        convertor = EdgeImpulseConverter()
    elif to_format == LabelFormat.YOLOV5:
        convertor = YoloV5Converter()
    elif to_format == LabelFormat.CVAT_XML:
        convertor = CVATXmlConverter()
    else:
        print('Unsupported output format {}'.format(to_format))

    for _ in range(100):
        random.shuffle(filtered)

    folder_prefix = "train_over_1k"
    # Write 100 by
    from_id = 0
    # to_id = 500
    #
    # folder_id = 20
    # while folder_id < 25:
    # for folder_id in range(25):
    to_id = 100

    folder_id = 0
    while folder_id < 10:
        chunk = filtered[from_id:to_id]
        print("Chunk is ", len(chunk))
        label_counts = count_labels(chunk)
        id = 0
        for key, val in label_counts.items():
            print("\t", id, key, val)
            id += 1

        # path_out = os.path.join(os.path.dirname(path_in),
        path_out=os.path.join(path_in,
                              "{}_{}.xml".format(folder_prefix, folder_id))

        project_name = "{}_{}".format(folder_prefix, folder_id)
        # convertor.write(project_name, chunk, path_out)
        convertor.write(project_name, chunk, path_out)

        # # move all data files
        # print(f"#project_name: {project_name} {path_in}")
        move_data_files(path_in, project_name, chunk)

        from_id = to_id + 1
        to_id = from_id + 100

        folder_id += 1


def move_data_files(parent_folder, folder_to, chunk):
    moved_count = 0

    folder_to = os.path.join(parent_folder, folder_to)
    if not os.path.exists(folder_to):
        print("Creating folder to ", folder_to)
        os.mkdir(folder_to)

    for image_info in chunk:
        image_file = image_info[0]
        folder_from = os.path.join(parent_folder, image_info[4])

        image_file = os.path.join(folder_from, image_file)
        txt_file = os.path.join(folder_from, image_file[:-3] + "txt")

        if os.path.exists(image_file):
            dest_image = os.path.join(folder_to, os.path.basename(image_file))

            if os.path.exists(dest_image):
                print("ERROR: target {} already exists".format(dest_image))
                exit(-1)

            shutil.copy(image_file, dest_image)
            moved_count += 1
        else:
            print("ERROR: {} does not exist".format(image_file))
            exit(-1)

        if os.path.exists(txt_file):
            dest_txt = os.path.join(folder_to, os.path.basename(txt_file))

            if os.path.exists(dest_txt):
                print("ERROR: target {} already exists".format(dest_txt))
                exit(-1)

            shutil.copy(txt_file, dest_txt)
            moved_count += 1
        else:
            print("ERROR: source {} does not exist".format(txt_file))
            exit(-1)

    print("Moved ", moved_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", action="store", type=Mode.argparse, choices=list(Mode), dest="mode")
    parser.add_argument("--format_in", action="store", type=LabelFormat.argparse, choices=list(LabelFormat), dest="format_in")
    parser.add_argument("--format_out", action="store", type=LabelFormat.argparse, choices=list(LabelFormat), dest="format_out")
    parser.add_argument("--path_in", action="store", dest="path_in", type=str)
    parser.add_argument("--path_out", action="store", dest="path_out", type=str)

    args = parser.parse_args()
    print(args.mode)

    if args.mode == Mode.FILTER:
        filter_files(args.path_in, args.format_in, args.format_out)


    else:
        print("Please specify the mode")
