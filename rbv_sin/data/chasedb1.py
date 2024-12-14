
from pathlib import Path

import numpy as np
import skimage.io

from .general import RFI, RFISet
from .loader import DatasetLoader

class ChaseDB1Loader(DatasetLoader):
    """
    Loader class for the ChaseDB1 retinal fundus image dataset for blood vessel
    segmentation. This class facilitates loading and translation of ChaseDB1 data into
    our general data format for further processing.
    """

    def __init__(self, root_path : Path) -> None:
        """
        Initialises the ChaseDB1 loading construct. It goes through the ChaseDB1 file hierarchy and
        sets up the information about all samples, however, it does not load images.

        Arguments:
        - 'root_path' - Path to the root of FIVES file hierarchy as downloaded.
        """
        super().__init__(root_path)

    def _create(self) -> None:
        """
        Creates descriptions of samples and groups from the ChaseDB1 file hierarchy.
        """
        self.descriptions = []

        files = [f if f.is_file() else None for f in self.root_path.iterdir()]
        for f in files:
            if f is None or f.suffix != ".jpg":
                continue
            value = {
                "image" : f,
                "stem" : f.stem,
                "gt_1" : Path(f.parent, f.stem + "_1stHO.png"),
                "gt_2" : Path(f.parent, f.stem + "_2ndHO.png"),
            }
            self.descriptions.append(value)

        self.samples = [None] * len(self.descriptions)
        all_indices = np.arange(0, len(self.descriptions))
        self.group_indices = {"all" : all_indices}

    def load(self, idx : int) -> None:
        """
        Loads the images and masks of the sample at the requested index. This function uses
        the description of a sample generated in the initialisation of the loader to find the correct
        files and load them.

        Arguments:
        - 'idx' - Index of the loaded sample. It is the index from the entire dataset, not just one subset.
        """
        if self.samples[idx] is not None:
            return
        description = self.descriptions[idx]
        sample = RFI(description["stem"])
        sample.image = skimage.io.imread(description["image"])
        sample.vessel_mask = [skimage.io.imread(description["gt_1"]), skimage.io.imread(description["gt_2"])]
        self.samples[idx] = sample

    def getDatasetName(self) -> str:
        return "Chase DB1"
