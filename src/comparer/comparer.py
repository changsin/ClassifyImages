#
# Copyright (C) 2020 TestWorks Inc.
#
# Author: changsin@
#

from abc import ABC, abstractmethod


class Comparer(ABC):

    @classmethod
    def version(cls): return "1.0"

    @abstractmethod
    def compare(self, image1, image2):
        raise NotImplementedError
