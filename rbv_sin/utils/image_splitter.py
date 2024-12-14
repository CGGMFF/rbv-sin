
from typing import Tuple, Union
import numpy as np

class ImageSplitter:
    """Class implementing the splitting of images into tiles."""

    def __init__(self, target_shape : Union[int, Tuple[int, int]], stride : Union[int, Tuple[int, int]]) -> None:
        """
        Initialises the image splitter.

        Arguments:
        - 'target_shape' - The target shape of created patches.
        - 'stride' - The stride of the created patches.
        """
        self.target_shape = (target_shape, target_shape) if isinstance(target_shape, int) else target_shape
        self.stride = (stride, stride) if isinstance(stride, int) else stride

    def split(self, image : np.ndarray) -> np.ndarray:
        """
        Execute the image splitting. It will generate an array of stacked patches of the same type as the image.

        Arguments:
        - 'image' - The source image.

        Returns:
        - An array of stacked patches extracted from the image.
        """
        patches = []
        for row in range(0, image.shape[0] - self.target_shape[0] + 1, self.stride[0]):
            for column in range(0, image.shape[1] - self.target_shape[1] + 1, self.stride[1]):
                patches.append(image[row : row + self.target_shape[0], column : column + self.target_shape[1]])
            if column < image.shape[1] - self.target_shape[1]:
                patches.append(image[row : row + self.target_shape[0], image.shape[1] - self.target_shape[1] :])
        if row < image.shape[0] - self.target_shape[0]:
            for column in range(0, image.shape[1] - self.target_shape[1] + 1, self.stride[1]):
                patches.append(image[image.shape[0] - self.target_shape[0]:, column : column + self.target_shape[1]])
            if column < image.shape[1] - self.target_shape[1]:
                patches.append(image[image.shape[0] - self.target_shape[0]:, image.shape[1] - self.target_shape[1] :])
        patches = np.asarray(patches)
        return patches
    
    def _insertPatch(self, image : np.ndarray, counts : np.ndarray, patches : np.ndarray, patch_idx : int, rs : int, re : int, cs : int, ce : int) -> int:
        """
        Inserts a patch into an image during composition. This is a helper function for reducing the code size and duplication.

        Arguments:
        - 'image' - The composed image of summed patch contributions.
        - 'counts' - The patch contribution counter.
        - 'patches' - The array of patches from which the iamge is being composed.
        - 'patch_idx' - The index of the inserted patch.
        - 'rs' - Row start index.
        - 're' - Row end index.
        - 'cs' - Column start index.
        - 'ce' - Column end index.

        Returns:
        - Incremented patch index.
        """
        image[rs : re, cs : ce] += patches[patch_idx]
        counts[rs : re, cs : ce] += np.ones(patches[patch_idx].shape[:2])
        return patch_idx + 1

    def compose(self, patches : np.ndarray, original_shape : Tuple[int, ...]) -> np.ndarray:
        """
        Composes an image from a set of patches - it will reconstruct the image exactly if the patches were created with 'split'.

        Arguments:
        - 'patches' - The array of patches for image reconstruction.
        - 'original_shape' - The shape of the image, which should be reconstructed.

        Returns:
        - The composed, reconstructed image.
        """
        counts = np.zeros(original_shape[:2])
        image = np.zeros(original_shape)
        patch_idx = 0
        for row in range(0, image.shape[0] - self.target_shape[0] + 1, self.stride[0]):
            for column in range(0, image.shape[1] - self.target_shape[1] + 1, self.stride[1]):
                patch_idx = self._insertPatch(image, counts, patches, patch_idx, row, row + self.target_shape[0], column, column + self.target_shape[1])
            if column < image.shape[1] - self.target_shape[1]:
                patch_idx = self._insertPatch(image, counts, patches, patch_idx, row, row + self.target_shape[0], image.shape[1] - self.target_shape[1], image.shape[1])
        if row < image.shape[0] - self.target_shape[0]:
            for column in range(0, image.shape[1] - self.target_shape[1] + 1, self.stride[1]):
                patch_idx = self._insertPatch(image, counts, patches, patch_idx, image.shape[0] - self.target_shape[0], image.shape[0], column, column + self.target_shape[1])
            if column < image.shape[1] - self.target_shape[1]:
                patch_idx = self._insertPatch(image, counts, patches, patch_idx, image.shape[0] - self.target_shape[0], image.shape[0], image.shape[1] - self.target_shape[1], image.shape[1])
        counts[counts == 0] = 1
        counts = np.expand_dims(counts, -1) if image.ndim > counts.ndim else counts
        result = image / counts
        return result
