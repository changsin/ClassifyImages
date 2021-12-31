from enum import Enum

IMAGE_SIZE = 320


class Mode(Enum):
    CONVERT = 'convert'
    FILTER = 'filter'
    REMOVE_UNLABELED_FILES = 'remove_unlabeled_files'

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
                    "warning@TPMS", "warning@Tire", "warning@Washer"]
DB_EXCLUDE = [
    # "alert@Retaining",
    # "alert@Distance",
    # "warning@ABS",
    # "alert@Coolant",
    # "warning@Fuel",
    "warning@Retaining",
    "warning@Steering",
    "alert@EngineOilTemp",
    "warning@Glow",
    "warning@CentralMonitoring",
    "warning@EPC",
    "warning@Washer"
]
DB_TOPS = [
    # Top 5
    "alert@Seatbelt",
    "warning@Engine",
    "alert@Parking",
    "warning@Tire",
    "warning@StabilityOn",

    # Top 10
    "alert@Brake",
    "warning@StabilityOff",
    "warning@Brake",
    "alert@Steering",
    "warning@Parking",

    # Top 15
    "alert@Retaining",
    "alert@Distance",
    "warning@ABS",
    "alert@Coolant",
    "warning@Fuel"
]
DB_TOP5 = DB_TOPS[:5]
DB_TOP10 = DB_TOPS[5:10]
DB_TOP15 = DB_TOPS[10:15]
SIDEWALK_CLASSES = [
    "wheelchair", "truck", "tree_trunk", "traffic_sign", "traffic_light",
    "traffic_light_controller", "table", "stroller", "stop", "scooter",
    "potted_plant", "power_controller", "pole", "person", "parking_meter",
    "movable_signage", "motorcycle", "kiosk", "fire_hydrant", "dog",
    "chair", "cat", "carrier", "car", "bus",
    "bollard", "bicycle", "bench", "barricade"]
SW_EX1 = ["bus", "truck", "person", "car"]
SW_EXCLUDE = [
    "tree_trunk", "traffic_sign", "traffic_light",
    "potted_plant", "pole",
    "movable_signage", "motorcycle",
    "chair",
    "bollard", "bicycle", "bench"]
SW_TOP15 = [
            "bench", "chair", "bus", "bicycle", "motorcycle",
            "potted_plant", "movable_signage", "truck", "traffic_light", "traffic_sign",
            # "bollard", "pole", "person", "tree_trunk", "car"
        ]
# SW_TOP15 = [
#     "truck", "tree_trunk", "traffic_sign", "traffic_light", # "wheelchair",
#     # "traffic_light_controller", "table", "stroller", "stop", "scooter",
#     "potted_plant", "pole", "person",       # "power_controller", "parking_meter",
#     "movable_signage", "motorcycle",        # "kiosk", "fire_hydrant", "dog",
#     "chair", "car", "bus",                  # "cat", "carrier"
#     "bollard", "bicycle", "bench",          #"barricade",
#         ]
SW_IGNORE = [
        "barricade",
        "carrier",
        "cat",
        "dog",
        "fire_hydrant",
        "kiosk",
        "parking_meter",
        "power_controller",
        "table",
        "traffic_light_controller",
        "scooter",
        "stop",
        "stroller",
        "wheelchair",
    ]


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