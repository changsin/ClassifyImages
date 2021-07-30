#
# Copyright (C) 2020 TestWorks Inc.
#
# Author: changsin@
#

import logging
import os

import cv2
import pytest

from src import HistogramComparer

# TODO: do a proper package-wide config setting later
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TEST_DATA_FOLDER = os.getcwd() + "/../data/images/"


def test_rgb_compare():
    image1 = cv2.imread(TEST_DATA_FOLDER + "tedworks.png")
    assert image1 is not None

    comparer = HistogramComparer()
    similarities = comparer.compare(image1, image1)
    logger.info("Similarities are: %d",  similarities)
    assert similarities == 100.00

    image_fake = cv2.imread(TEST_DATA_FOLDER + "testworks-screenshot.png")
    similarities = comparer.compare(image1, image_fake)
    logger.info("Similarities are: %d",  similarities)
    assert similarities < 75.00


if __name__ == '__main__':
    pytest.main()
