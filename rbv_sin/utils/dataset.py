
from typing import Tuple, Sequence, Union
from pathlib import Path

import numpy as np

from rbv_sin.data import RFISet
from rbv_sin.utils import transform

def applyRoi(rfiset : RFISet, roi_path : Path) -> Tuple[RFISet, np.ndarray]:
    """
    Crops the data from the source RFISet to some region of interest described in a text file in
    the directory 'roi_path'.

    Arguments:
    - 'rfiset' - RFISet consisitng of samples which should be cropped to some region ofi nterest.
    - 'roi_path' - Path towards a directory containing text file roi descriptions for samples from the 'rfiset'.

    Returns:
    - A new RFISet consisting of samples cropped to their regions of interest.
    - Numpy array of the region of interest coordinates loaded from the target path.
    """
    cropped_rfis = []
    rois = []
    for rfi in rfiset:
        roi_file = Path(roi_path, rfi.name + ".txt")
        with open(roi_file, "r") as file:
            line = file.readlines()[0].split()
        roi = (int(line[0]), int(line[1]), int(line[2]), int(line[3]))
        rois.append(roi)
        crop = transform.RFICrop((roi[1], roi[3], roi[0], roi[2], 0, 0))
        cropped_rfis.append(crop.apply(rfi))
    return RFISet(rfiset.name, cropped_rfis), np.asarray(rois, int)

class SampleFilter:
    """Class realising dataset filtering."""

    def removeIndices(self, data : RFISet, indices : Union[int, np.ndarray]) -> RFISet:
        """
        Removes the samples according to the given indices.

        Arguments:
        - 'data' - RFISet dataset for filtering.
        - 'indices' - Indices of samples, which should be removed from the dataset.

        Returns:
        - The filtered dataset.
        """
        selection = np.ones(len(data), dtype=int)
        selection[indices] = 0
        filtered_set = []
        for selected, sample in zip(selection, data):
            if selected > 0:
                filtered_set.append(sample)
        return RFISet(data.name, filtered_set)

    def removeNames(self, data : RFISet, names : Sequence[str]) -> RFISet:
        """
        Removes the samples according to the given names.

        Arguments:
        - 'data' - RFISet dataset for filtering.
        - 'indices' - Names of samples, which should be removed from the dataset.

        Returns:
        - The filtered dataset.
        """
        filtered_set = []
        for sample in data:
            if sample.name not in names:
                filtered_set.append(sample)
        return RFISet(data.name, filtered_set)
