
from typing import Tuple, Union, List
import numpy as np
import skimage.morphology
import skimage.transform

from rbv_sin.utils.focus_mask import MaskBlobs
    
class MaskExtender:
    """Base class for mask extension."""

    def __init__(self) -> None:
        pass

    def __call__(self, mask : np.ndarray) -> np.ndarray:
        """Virtual method for mask extension application."""
        raise NotImplementedError()

class MaskMorphology(MaskExtender):
    """Mask extender which uses morphology to extend the masks."""

    def __init__(self, radius : int, mode : str = "dilate", threshold : float = 0.5) -> None:
        """
        Initalises the morphological mask extender.

        Arguments:
        - 'radius' - The radius of disc structuring element used in morphology.
        - 'mode' - The morphology mode, either 'dilation' or 'erosion'.
        - 'threshold' - Threshold applied before morphology.
        """
        super().__init__()
        self.radius = radius
        self.mode = mode
        self.threshold = threshold

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        """Applies the morphological operation to the 'mask'."""
        if self.mode in ["dilate", "dilation"]:
            return skimage.morphology.dilation(mask > self.threshold, skimage.morphology.disk(self.radius))
        elif self.mode in ["erode", "erosion"]:
            return skimage.morphology.erosion(mask > self.threshold, skimage.morphology.disk(self.radius))
        else:
            raise ValueError("Unknown morphology mode: {}.".format(self.mode))

class VesselDataGenerator:
    """Generator for false blood vessels."""

    DEFAULT_ANGLE_RANGE = (40, 320)

    def __init__(self, angle_range : Tuple[float, float] = None, rng : Union[int, np.random.RandomState] = None) -> None:
        """
        Initialises the false blood vessel generator.

        Arguments:
        - 'angle_range' - The angle range for random vessel rotation.
        - 'rng' - The random number genrator for deterministic generation.
        """
        self.angle_range = VesselDataGenerator.DEFAULT_ANGLE_RANGE if angle_range is None else angle_range
        self.generator = rng if isinstance(rng, np.random.RandomState) else np.random.RandomState(rng)

    def generate(self, vessel_mask : np.ndarray, count : int = 1, threshold : float = 0.5) -> List[np.ndarray]:
        """
        Generates a false blood vessel mask by randomly rotating the real mask.

        Arguemnts:
        - 'vessel_mask' - The true blood vessel mask.
        - 'count' - The number of generated masks.
        - 'threshold' - The threshold applied to the mask before transformation.

        Returns:
        - The list of generated masks.
        """
        vessel_mask = vessel_mask > threshold
        generated_masks = []
        for _ in range(count):
            rnd_angle = self.generator.random() * (self.angle_range[1] - self.angle_range[0]) + self.angle_range[0]
            rotated_mask = skimage.transform.rotate(vessel_mask, rnd_angle)
            generated_masks.append(rotated_mask)
        return generated_masks

class CustomMaskGenerator:
    """Generator for custom training masks for the inpainting network."""

    def __init__(self, target_shape : Tuple[int, int], blob_radius_fraction : float = 0.25, angle_range : Tuple[float, float] = None, blob_density : float = 0.5,
                 gen_mask_extender : MaskExtender = None, src_mask_extender : MaskExtender = None, rng : Union[int, np.random.RandomState] = None) -> None:
        """
        Initialises the custom training mask generator. It is composed of a transformation of a real blood vessel mask
        and a focus mask for improvement of training capabilities in the ONH region.

        Arguments:
        - 'target_shape' - The shape of the generated mask.
        - 'blob_radius_fraction' - The fraction of the mask height which is used as the radius for focus mask generation.
        - 'angle_range' - The angle range for random rotation of the true blood vessel mask.
        - 'blob_density' - The covered percentage of the focus mask.
        - 'gen_mask_extender' - Mask extension applied on the generated masks.
        - 'src_mask_extender' - Mask extension applied on the true (source) vessel masks before multiplication with the generated vessels.
        - 'rng' - The random number generator for deterministic generation.
        """
        self.rng = rng if isinstance(rng, np.random.RandomState) else np.random.RandomState(rng)
        self.vessel_generator = VesselDataGenerator(angle_range, self.rng)
        self.blob_generator = MaskBlobs(target_shape, int(blob_radius_fraction * target_shape[0]), self.rng)
        self.gen_mask_extender = (lambda x : x) if gen_mask_extender is None else gen_mask_extender
        self.src_mask_extender = (lambda x : x) if src_mask_extender is None else src_mask_extender
        self.generated_count = 1
        self.mask_threshold = 0.5
        self.blob_density = blob_density

    def _shiftMask(self, mask : np.ndarray, centre : Tuple[float, float]) -> np.ndarray:
        """Private method which shifts the 'mask' to the given 'centre'."""
        mask_centre = (mask.shape[0] / 2, mask.shape[1] / 2)
        diff = (int(centre[0] - mask_centre[0]), int(centre[1] - mask_centre[1]))
        pad_rt, pad_rb, pad_cl, pad_cr = max(diff[0], 0), np.abs(min(diff[0], 0)), max(diff[1], 0), np.abs(min(diff[1], 0))
        padded = np.pad(mask, [[pad_rt, pad_rb], [pad_cl, pad_cr]])
        cropped = padded[pad_rb : pad_rb + mask.shape[0], pad_cr : pad_cr + mask.shape[1]]

        return cropped
    
    def generateMask(self, vessel_mask : np.ndarray, optic_centre : Tuple[float, float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates the custom training mask from the given source true blood vessel mask and the optic centre.

        Arguments:
        - 'vessel_mask' - The trasnformed true blood vessel mask.
        - 'optic_centre' - The optic centre where the focus mask should be generated.

        Returns:
        - The final generated training mask.
        - The combination of transformed vessel and focus masks.
        - The generated focus mask.
        - The transformed blood vessel mask. 
        """
        transformed_mask = self.vessel_generator.generate(vessel_mask, self.generated_count, self.mask_threshold)[0]
        focus_mask = self.blob_generator.generate(self.generated_count, self.blob_density)[0]
        if optic_centre is not None:
            focus_mask = self._shiftMask(focus_mask, optic_centre)

        transformed_mask = self.gen_mask_extender(transformed_mask)
        focus_mask = self.gen_mask_extender(focus_mask)

        combined_mask = focus_mask + (1.0 - focus_mask) * transformed_mask
        final_mask = (1.0 - self.src_mask_extender(vessel_mask)) * combined_mask

        return final_mask, combined_mask, focus_mask, transformed_mask
