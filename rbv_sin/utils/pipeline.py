
from typing import Union, Sequence, List, Tuple
import numpy as np

from rbv_sin.utils.interfaces import SINSegmentation, SINInpainting
from rbv_sin.utils.mask_gen import MaskExtender

class SINPipeline:
    """Implements the segmentation-to-inpainting pipeline for blind blood vessel inpainting."""

    def __init__(self, segmentation_method : SINSegmentation, inpainting_method : SINInpainting, mask_extender : MaskExtender) -> None:
        """
        Initalises the blind vessel inpainting pipeline.

        Arguments:
        - 'segmentation_method' - A segmentation algorithm.
        - 'inpainting_method' - An inpainting algorithm.
        - 'mask_extender' - Mask extension applied on the segmented mask before inpainting.
        """
        self.segmentation_method = segmentation_method
        self.inpainting_method = inpainting_method
        self.mask_extender = mask_extender

    def _evaluate(self, image : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Private execution of the pipeline."""
        vessel_mask = self.segmentation_method(image)
        extended_vessel_mask = self.mask_extender(vessel_mask)
        inpainted = self.inpainting_method(image, extended_vessel_mask)
        return inpainted, vessel_mask

    def evaluate(self, image : Union[np.ndarray, Sequence[np.ndarray], None], is_sequence : bool = False) -> Union[Tuple[np.ndarray, np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Evaluation of the pipeline.
        Blood vessel mask is segmented using the segmentation algorithm. Then the mask is extended
        with the mask extension algorithm and finally the extended mask is inpainted.

        Arguemnts:
        - 'image' - The image to be blindly inpainted.
        - 'is_sequence' - Whether 'image' is a sequence of iamges for inpainting.

        Returns:
        - The inpainted image (or a list of images depending on 'is_sequence').
        """
        if image is None:
            raise ValueError("No image to process in the pipeline.")
        if image.ndim > 4:
            raise ValueError("The image has too many dimensions: {}.".format(image.ndim))
        is_sequence = is_sequence or image.ndim > 3
        if is_sequence:
            return [self._evaluate(single_image) for single_image in image]
        return self._evaluate(image)
