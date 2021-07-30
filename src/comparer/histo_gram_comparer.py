#
# Copyright (C) 2020 TestWorks Inc.
#
# 2020-02-18: changsin@ created
#

import logging

import cv2

from src import Comparer

# TODO: do a proper package-wide config setting later
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HistogramComparer(Comparer):

    @classmethod
    def version(cls): return "1.0"

    @classmethod
    def compare(cls, image1, image2):
        img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
        return round(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL) * 100, 2)
