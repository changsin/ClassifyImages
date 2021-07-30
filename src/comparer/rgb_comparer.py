#
# Copyright (C) 2020 TestWorks Inc.
#
# 2020-02-18: changsin@ created
#

import logging

import numpy as np

from src import Comparer

# TODO: do a proper package-wide config setting later
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RGBComparer(Comparer):

    @classmethod
    def version(cls): return "1.0"

    @classmethod
    def compare(cls, image1, image2):
        diff_arr = image1.astype(np.int32) - image2.astype(np.int32)

        non_matching_pixels = np.count_nonzero(diff_arr > 0)
        matching_pixels = np.count_nonzero(diff_arr == 0)

        total_pixels = non_matching_pixels + matching_pixels
        logger.info("total_pixels and non-matching: %d/%d", non_matching_pixels, total_pixels)

        return round((total_pixels - non_matching_pixels) / total_pixels * 100, 2)
