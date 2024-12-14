
from pathlib import Path
from typing import List, Dict, Tuple
import re

import numpy as np
import skimage.io
import skimage.measure
import skimage.morphology
import cv2 as cv

from .general import RFI, RFISet
from .loader import DatasetLoader
from .utils import meanMask

class RigaLoader(DatasetLoader):

    _PRIME = Path("prime")
    _CONNECTION = Path("-")
    _EXPERTS = 6
    _THRESHOLD_OFFSET = 5

    def __init__(self, root_path: Path) -> None:
        """
        Initialises the RIGA subset loading construct. It goes through the RIGA subset file hierarchy and
        sets up the information about all samples but it does not load images.

        Arguments:
        - 'root_path' - Path to the root of RIGA subset file hierarchy as downloaded.
        """
        super().__init__(root_path)

    def _getDiffMask(self, image : np.ndarray, gt : np.ndarray, retina_mask : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the mask of pixels with a significant difference from ground truth.

        Arguments:
        - 'image' - The image for difference computation.
        - 'gt' - The ground truth for the difference computation.
        - 'retina_mask' - The mask of valid pixels in the retina.

        Returns:
        - The difference mask.
        - The difference image (before the thresholding).
        """
        diff = np.abs(gt.astype(int) - image.astype(int))[:, :, 0] * retina_mask
        diff_mask = diff > 1
        return diff_mask, diff
    
    def _countLabels(self, diff_mask : np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Computes the connected component labeling and a sorted array of pixel counts of the components.

        Arguments:
        - 'diff_mask' - The mask for connected component computation.

        Returns:
        - The labeled image where each connected component has a unique label.
        - The sorted list of pixel area containing a tuple for each (index, area) for each component.
        """
        num, labeled, stats, centroids = cv.connectedComponentsWithStats(diff_mask.astype(np.uint8))
        counts = [(idx, stats[idx, cv.CC_STAT_AREA]) for idx in range(1, num)]
        counts = sorted(counts, key=lambda x: x[1], reverse=True)
        if len(counts) < 1:
            raise ValueError("Image in a RIGA-like dataset has no distinguishable contour.")
        return labeled, counts
    
    def _loadMasks(self, image : np.ndarray, gt : np.ndarray, retina_mask : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the optic disc and optic cup masks from a ground truth image and an image with
        object boundaries marked by a different colour.

        Arguments:
        - 'image' - The source image with marked boundaries.
        - 'gt' - The ground truth image without boundaries.
        - 'retina_mask' - The mask of valid retinal pixels.

        Returns:
        - The optic disc mask.
        - The optic cup mask.
        """
        # OD
        diff_mask, diff = self._getDiffMask(image, gt, retina_mask)
        labeled, counts = self._countLabels(diff_mask)
        od_label = counts[0][0]
        contour_od = np.argwhere(labeled == od_label)[:, [1, 0]]
        hull_od = cv.convexHull(contour_od)
        mask_od = np.zeros(diff_mask.shape[:2], dtype=np.uint8)
        mask_od = cv.fillPoly(mask_od, [hull_od], 255)
        # OC
        kernel = skimage.morphology.disk(8)
        od_eroded = cv.erode(mask_od, kernel, iterations = 1)
        diff_mask = diff_mask * od_eroded > 0
        labeled, counts = self._countLabels(diff_mask)
        oc_label = counts[0][0]
        contour_oc = np.argwhere(labeled == oc_label)[:, [1, 0]]
        hull_oc = cv.convexHull(contour_oc)
        mask_oc = np.zeros(labeled.shape[:2], dtype=np.uint8)
        mask_oc = cv.fillPoly(mask_oc, [hull_oc], 255)
        return mask_od, mask_oc
    
    def _getRetinaMask(self, sample_image : np.ndarray) -> np.ndarray:
        """
        Computes the mask of valid retinal pixels in the image - nonzero pixels with a threhsold.

        Arguments:
        - 'sample_image' - The retinal image where we search for valid pixels.

        Returns:
        - The mask of valid pxiels.
        """
        mean_image = np.mean(sample_image, axis=2)
        retina_mask = np.logical_and(mean_image >= RigaLoader._THRESHOLD_OFFSET, mean_image <= 255 - RigaLoader._THRESHOLD_OFFSET)
        kernel = skimage.morphology.disk(16)
        retina_mask = cv.morphologyEx(cv.morphologyEx(retina_mask.astype(np.uint8), cv.MORPH_CLOSE, kernel), cv.MORPH_OPEN, kernel)
        retina_mask[:, 0] = 0
        retina_mask[:, -1] = 0
        retina_mask[0, :] = 0
        retina_mask[-1, :] = 0
        retina_mask = cv.erode(retina_mask, kernel, iterations = 1)
        return retina_mask
    
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
        sample.od_mask = []
        sample.oc_mask = []
        retina_mask = self._getRetinaMask(sample.image)
        for mask_image_path in description["gts"]:
            mask_image = skimage.io.imread(mask_image_path)
            od_mask, oc_mask = self._loadMasks(sample.image, mask_image, retina_mask)
            sample.od_mask.append(od_mask > 0)
            sample.oc_mask.append(oc_mask > 0)
        sample.mean_od_mask = meanMask(sample.od_mask) > 0.5
        sample.mean_oc_mask = meanMask(sample.oc_mask) > 0.5
        self.samples[idx] = sample

class MessidorLoader(RigaLoader):
    """
    Loader class for the MESSIDOR retinal fundus image dataset from the RIGA set for optic nerve head
    segmentation. This class facilitates loading and translation of MESSIDOR data into
    our general data format for further processing.
    """

    def __init__(self, root_path: Path) -> None:
        """
        Initialises the MESSIDOR loading construct. It goes through the MESSIDOR file hierarchy and
        sets up the information about all samples but it does not load images.

        Arguments:
        - 'root_path' - Path to the root of MESSIDOR file hierarchy as downloaded.
        """
        super().__init__(root_path)

    def _getRetinaMask(self, sample_image : np.ndarray) -> np.ndarray:
        """
        Computes the mask of valid retinal pixels.

        Arguments:
        - 'sample_image' - The retinal image.

        Returns:
        - The mask of valid pixels.
        """
        mean_image = np.mean(sample_image, axis=2)
        retina_mask = np.logical_and(mean_image >= RigaLoader._THRESHOLD_OFFSET, mean_image <= 255 - RigaLoader._THRESHOLD_OFFSET)
        return retina_mask.astype(np.uint8)

    def _create(self) -> None:
        """
        Creates descriptions of samples and groups from the MESSIDOR file hierarchy.
        """
        self.base_path = Path(self.root_path)
        self.descriptions = []

        def _collectGroup(image_path : Path, gt_experts : int) -> List[Dict[str, str]]:
            pattern = re.compile("\Aimage.*{}.tif".format(MessidorLoader._PRIME))
            prime_files : List[Path] = []
            for f in image_path.iterdir():
                if f.is_file() and pattern.fullmatch(f.name):
                    prime_files.append((int(f.stem[5:-5]), f))
            prime_files = sorted(prime_files, key=lambda x: x[0])
            group = []
            for _, f in prime_files:
                value = {
                    "image" : f,
                    "stem" : f.stem[:-5],
                    "gts" : [Path(image_path, f.stem[:-5] + "{}{}.tif".format(MessidorLoader._CONNECTION, exp + 1)) for exp in range(gt_experts)]
                }
                group.append(value)
            return group
        
        group = _collectGroup(self.base_path, MessidorLoader._EXPERTS)
        self.descriptions.extend(group)
        self.samples = [None] * len(self.descriptions)
        all_indices = np.arange(0, len(self.descriptions))
        self.group_indices = {"all" : all_indices}

class MagrabiaLoader(RigaLoader):
    """
    Loader class for the MAGRABIA retinal fundus image dataset from the RIGA set for optic nerve head
    segmentation. This class facilitates loading and translation of MAGRABIA data into
    our general data format for further processing.
    """

    _MALE_RELPATH = "MagrabiaMale"
    _FEMALE_RELPATH = "MagrabiFemale"

    def __init__(self, root_path: Path) -> None:
        """
        Initialises the MAGRABIA loading construct. It goes through the MAGRABIA file hierarchy and
        sets up the information about all samples but it does not load images.

        Arguments:
        - 'root_path' - Path to the root of MAGRABIA file hierarchy as dwnloaded.
        """
        super().__init__(root_path)

    def _create(self) -> None:
        """
        Creates descriptions of samples and groups from the MAGRABIA file hierarchy.
        """
        self.base_path = Path(self.root_path)
        self.descriptions = []

        def _collectGroup(image_path : Path, gt_experts : int) -> List[Dict[str, str]]:
            pattern = re.compile("\Aimage.*{}.tif".format(MagrabiaLoader._PRIME))
            prime_files : List[Path] = []
            for f in image_path.iterdir():
                if f.is_file() and pattern.fullmatch(f.name):
                    prime_files.append((int(f.stem[5:-5]), f))
            prime_files = sorted(prime_files, key=lambda x: x[0])
            group = []
            for _, f in prime_files:
                value = {
                    "image" : f,
                    "stem" : f.stem[:-5],
                    "gts" : [Path(image_path, f.stem[:-5] + "{}{}.tif".format(MagrabiaLoader._CONNECTION, exp + 1)) for exp in range(gt_experts)]
                }
                group.append(value)
            return group
        
        self.male_path = Path(self.base_path, MagrabiaLoader._MALE_RELPATH)
        self.female_path = Path(self.base_path, MagrabiaLoader._FEMALE_RELPATH)
        counts = []
        for set_path in [self.male_path, self.female_path]:
            group = _collectGroup(set_path, MagrabiaLoader._EXPERTS)
            counts.append(len(group))
            self.descriptions.extend(group)
        self.samples = [None] * len(self.descriptions)
        male_indices = np.arange(0, counts[0])
        female_indices = np.arange(counts[0], np.sum(counts))
        all_indices = np.arange(0, np.sum(counts))
        self.group_indices = {"male" : male_indices, "female" : female_indices, "all" : all_indices}

class BinRushedLoader(RigaLoader):
    """
    Loader class for the BinRushed retinal fundus image dataset from the RIGA set for optic nerve head
    segmentation. This class facilitates loading and translation of BinRushed data into
    our general data format for further processing.
    """

    _GROUP_1_RELPATH = "BinRushed1"
    _GROUP_2_RELPATH = "BinRushed2"
    _GROUP_3_RELPATH = "BinRushed3"
    _GROUP_4_RELPATH = "BinRushed4"

    def __init__(self, root_path: Path) -> None:
        """
        Initialises the BinRushed loading construct. It goes through the BinRushed file hierarchy and
        sets up the information about all samples but it does not load images.

        Arguments:
        - 'root_path' - Path to the root of BinRushed file hierarchy as downloaded.
        """
        super().__init__(root_path)

    def _getDiffMask(self, image : np.ndarray, gt : np.ndarray, retina_mask : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the difference mask between an image with marked boundaries and a ground truth image.

        Arguments:
        - 'image' - The image with marked ONH boundaris.
        - 'gt' - The ground truth image without boundaries.
        - 'retina_mask' - The mask of valid retinal pixels.

        Returns:
        - The mask of difference pixels from the ground truth.
        - The differences before thresholding.
        """
        diff = np.abs(gt.astype(int) - image.astype(int))[:, :, 0] * retina_mask
        diff_mask = diff > 10
        diff_mask = skimage.morphology.remove_small_objects(diff_mask, min_size=32)
        return diff_mask, diff

    def _create(self) -> None:
        """
        Creates descriptions of samples and groups from the BinRushed file hierarchy.
        """
        self.base_path = Path(self.root_path)
        self.descriptions = []

        def _collectGroup(image_path : Path, gt_experts : int) -> List[Dict[str, str]]:
            pattern = re.compile("\Aimage.*{}\..*".format(BinRushedLoader._PRIME))
            prime_files : List[Path] = []
            for f in image_path.iterdir():
                if f.is_file() and pattern.fullmatch(f.name):
                    prime_files.append((int(f.stem[5:-5]), f))
            prime_files = sorted(prime_files, key=lambda x: x[0])
            group = []
            allowed_Extensions = ["jpg", "tif"]
            for _, f in prime_files:
                gts = []
                for exp in range(gt_experts):
                    for extension in allowed_Extensions:
                        gt_path = Path(image_path, f.stem[:-5] + "{}{}.{}".format(BinRushedLoader._CONNECTION, exp + 1, extension))
                        if gt_path.exists():
                            gts.append(gt_path)
                value = {
                    "image" : f,
                    "stem" : f.stem[:-5],
                    "gts" : gts
                }
                group.append(value)
            return group
        
        group_paths = [Path(self.base_path, group_relpath) for group_relpath in [BinRushedLoader._GROUP_1_RELPATH, BinRushedLoader._GROUP_2_RELPATH, BinRushedLoader._GROUP_3_RELPATH, BinRushedLoader._GROUP_4_RELPATH]]
        counts = []
        for set_path in group_paths:
            group = _collectGroup(set_path, MagrabiaLoader._EXPERTS)
            counts.append(len(group))
            self.descriptions.extend(group)
        self.samples = [None] * len(self.descriptions)
        group_sums = np.cumsum([0] + counts)
        group_indices = [np.arange(group_sums[i], group_sums[i + 1]) for i in range(len(group_paths))]
        all_indices = np.arange(0, np.sum(counts))
        self.group_indices = {"all" : all_indices}
        for group_idx, indices in enumerate(group_indices):
            self.group_indices["bin_rushed_{}".format(group_idx + 1)] = indices
