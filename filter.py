from abc import ABC, abstractmethod


class Filter(ABC):

    @staticmethod
    def filter_by_label(parsed, label):
        filtered = []
        for image_info in parsed:
            for box in image_info[-1]:
                if label == box[0]:
                    print("Adding ", image_info[0])
                    filtered.append(image_info)
                    break

        return filtered

