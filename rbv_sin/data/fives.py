
from pathlib import Path
from typing import List, Dict

import numpy as np
import skimage.io

from .general import RFI, RFISet
from .loader import DatasetLoader

class FivesLoader(DatasetLoader):
    """
    Loader class for the FIVES retinal fundus image dataset for blood vessel
    segmentation. This class facilitates loading and translation of FIVES data into
    our general data format for further processing.
    """

    _TRAIN_RELPATH = Path("train")
    _TEST_RELPATH = Path("test")
    _IMAGE_RELPATH = Path("Original")
    _GT_RELPATH = Path("Ground truth")

    def __init__(self, root_path: Path) -> None:
        """
        Initialises the FIVES loading construct. It goes through the FIVES file hierarchy and
        sets up the information about all samples, however, it does not load images.

        Arguments:
        - 'root_path' - Path to the root of FIVES file hierarchy as dwnloaded.
        """
        super().__init__(root_path)

    def _create(self) -> None:
        """
        Creates descriptions of samples and groups from the FIVES file hierarchy.
        """
        self.descriptions = []

        def _collectGroup(image_path : Path, gt_path : Path) -> List[Dict[str, str]]:
            files = [f if f.is_file() else None for f in image_path.iterdir()]
            group = []
            for f in files:
                if f is None or f.suffix != ".png":
                    continue
                value = {
                    "image" : f,
                    "stem" : f.stem,
                    "gt" : Path(gt_path, f.name)
                }
                group.append(value)
            return group

        self.train_path = Path(self.root_path, FivesLoader._TRAIN_RELPATH)
        self.test_path = Path(self.root_path, FivesLoader._TEST_RELPATH)
        counts = []
        for set_path in [self.train_path, self.test_path]:
            image_path = Path(set_path, FivesLoader._IMAGE_RELPATH)
            gt_path = Path(set_path, FivesLoader._GT_RELPATH)
            group = _collectGroup(image_path, gt_path)
            counts.append(len(group))
            self.descriptions.extend(group)
        self.samples = [None] * len(self.descriptions)
        train_indices = np.arange(0, counts[0])
        test_indices = np.arange(counts[0], np.sum(counts))
        all_indices = np.arange(0, np.sum(counts))
        self.group_indices = {"train" : train_indices, "test" : test_indices, "all" : all_indices}

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
        sample.vessel_mask = skimage.io.imread(description["gt"])[:, :, 0]
        sample.vessel_mask = sample.vessel_mask
        self.samples[idx] = sample

    def getDatasetName(self) -> str:
        return "FIVES"
