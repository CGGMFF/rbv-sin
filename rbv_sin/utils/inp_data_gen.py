
from typing import Tuple, Sequence
import numpy as np
import skimage.filters

from rbv_sin.utils.mask_gen import CustomMaskGenerator

class OpticCentreEstimator:
    """
    Simple optic centre estimator. This implementation will not work well for full retinal images.
    It is good enough for OC localisation in an ONH region of interest.
    """

    def __init__(self, sigma : float = 13, channel : int = 0) -> None:
        """
        Initialises the optic centre estimator.

        Arguments:
        - 'sigma' - The gaussian sigma for filtration.
        - 'channel' - Which image channel is used for the OC estimation.
        """
        self.sigma = sigma
        self.channel = channel

    def __call__(self, image : np.ndarray) -> Tuple[float, float]:
        """
        Executes the optic centre localisation.

        Arguments:
        - 'image' - An image with optic disc.

        Returns:
        - Coordinates of the located optic centre.
        """
        bw = image[:, :, self.channel]
        filtered = skimage.filters.gaussian(bw, self.sigma)
        ind = np.unravel_index(np.argmax(filtered), filtered.shape)

        return (ind[0], ind[1])

class InpaintDataTransformer:
    """Transformer of data for the inpainting network."""

    def __init__(self, mask_generator : CustomMaskGenerator = None, centre_estimator : OpticCentreEstimator = None) -> None:
        """
        Initialises the trasnformer.

        Arguments:
        - 'mask_generator' - Generator for custom blood vessel inpainting mask.
        - 'centre_estimator' - If not None then it will be used to generate focus mask at the OC location.
        """
        self.mask_generator = mask_generator
        self.centre_estimator = centre_estimator

    def transform(self, image : np.ndarray, mask : np.ndarray, no_generation : bool = False) -> np.ndarray:
        """
        Applies the transformation to a single data sample.
        The transformation sets the masked image values to zero and concatenates the mask to the edited image.
        The mask is transformed to custom blood vessel inpainting training mask if 'mask_generator' is not None.

        Arguments:
        - 'image' - The transformed image.
        - 'mask' - The blood vessel mask.
        - 'no_generation' - Custom training mask will not be generated and 'mask' will be used directly.

        Returns:
        - Transformed image for the inpainting network inputs.
        """
        if self.mask_generator is None or no_generation:
            train_mask = mask
        else:
            train_mask, _, _, _ = self.mask_generator.generateMask(mask, None if self.centre_estimator is None else self.centre_estimator(image))
        train_image = image * np.expand_dims(1.0 - train_mask, -1)
        train_image = np.concatenate([train_image, np.expand_dims(train_mask, -1)], -1)
        return train_image

    def transformPairedBatch(self, batch : Sequence[Tuple[np.ndarray, np.ndarray]], no_generation : bool = False) -> np.ndarray:
        """
        Applies the transformation to a batch of samples.
        The transformation sets the masked image values to zero and concatenates the mask to the edited image.
        The mask is transformed to custom blood vessel inpainting training mask if 'mask_generator' is not None.

        Arguments:
        - 'batch' - The batch of images and blood vessel masks.
        - 'no_generation' - Custom training mask will not be generated and vessel masks will be used directly.

        Returns:
        - Array of the transformed samples.
        """
        train_images = []
        for _, (image, mask) in enumerate(batch):
            train_images.append(self.transform(image, mask, no_generation))
        return np.asarray(train_images)
