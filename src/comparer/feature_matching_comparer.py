#
# Copyright (C) 2020 TestWorks Inc.
#
# 2020-02-18: changsin@ created
#   implemented based on https://stackoverflow.com/questions/189943/how-can-i-quantify-difference-between-two-images
#

import logging

import cv2

from src import Comparer

# TODO: do a proper package-wide config setting later
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FeatureMatchingComparer(Comparer):

    @classmethod
    def version(cls): return "1.0"

    @classmethod
    def compare(cls, image1, image2):
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

        orb = cv2.ORB_create()
        # SIFT
        kp_1, des_1 = orb.detectAndCompute(image1, None)
        kp_2, des_2 = orb.detectAndCompute(image2, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # match.distance is a float between {0:100} - lower means more similar
        matches = bf.match(des_1, des_2)
        similar_regions = [i for i in matches if i.distance < 60]
        if len(matches) == 0:
            return 0

        logger.info("similar_regions/matches=%d/%d", len(similar_regions), len(matches))
        return round(len(similar_regions) / len(matches) * 100, 2)
