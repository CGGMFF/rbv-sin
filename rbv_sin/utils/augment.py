
from typing import Sequence, List, Union, Tuple
import time

import numpy as np
import skimage.transform

class Augmentor:
    """
    Class implementing custom sample and batch augmentations evaluated during training before running a single training step.
    The augmentation implemented in this class are optimised and significatn performance gains can be achieved by optimisation
    of these algorithm, especially their transfer to the GPU.
    """

    # Named constants for the types of augmentations.
    MIRROR_H = 0
    MIRROR_V = 1
    ROTATION = 2
    ZOOM = 3
    BRIGHTNESS = 4
    CONTRAST = 5

    # Default augmentation parameters.
    DEFAULT_ROT_BOUNDS = (-30, 30)
    DEFAULT_ZOOM_BOUNDS = (0.7, 1.3)
    DEFAULT_INTENSITY_RANGE = (0.5, 1.5)
    DEFAULT_CONTRAST_RANGE = (-1.0, 1.0)
    DEFAULT_AUGMENTATION_CHANCE = 0.5

    def __init__(self, rot_bounds : Tuple[float, float] = None, rot_count : int = None, zoom_bounds : Tuple[float, float] = None, zoom_count : int = None,
                 brightness_range : Tuple[float, float] = None, brightness_count : int = None, contrast_range : Tuple[float, float] = None, contrast_count : int = None,
                 rng : Union[int, np.random.RandomState] = None, aug_chance : float = None) -> None:
        """
        Initialises the augmentation object with the given augmentation parameters and random number generator.
        For batch augmentation, all counts should be left as None or set directly to 1. Larger counts can be used to generate data.

        Arguments:
        - 'rot_bounds' - The bounds of random rotation.
        - 'rot_count' - The number of augmented samples generated in 1 rotation augmentation.
        - 'zoom_bounds' - The bounds of random zoom.
        - 'zoom_count' - The number of augmented samples generated in 1 zoom augmentation.
        - 'brightness_range' - The bounds of random brightness.
        - 'brightness_count' - The number of augmented samples generated in 1 brightness augmentation.
        - 'contrast_range' - The bounds of random contrast.
        - 'contrast_count' - The number of augmented samples generated in 1 contrast augmentation.
        - 'rng' - Random number generator or a rng seed.
        - 'aug_chance' - Probability that a samples undergoes a particular augmentation.
        """
        self.rot_bounds = rot_bounds if rot_bounds is not None else Augmentor.DEFAULT_ROT_BOUNDS
        self.rot_count = rot_count if rot_count is not None else 1
        self.zoom_bounds = zoom_bounds if zoom_bounds is not None else Augmentor.DEFAULT_ZOOM_BOUNDS
        self.zoom_count = zoom_count if zoom_count is not None else 1
        self.brightness_range = brightness_range if brightness_range is not None else Augmentor.DEFAULT_INTENSITY_RANGE
        self.brightness_count = brightness_count if brightness_count is not None else 1
        self.contrast_range = contrast_range if contrast_range is not None else Augmentor.DEFAULT_CONTRAST_RANGE
        self.contrast_count = contrast_count if contrast_count is not None else 1
        self.augment_functions = {
            Augmentor.MIRROR_V : self._mirrorV,
            Augmentor.MIRROR_H : self._mirrorH,
            Augmentor.ROTATION : self._rotate,
            Augmentor.ZOOM : self._zoom,
            Augmentor.BRIGHTNESS : self._brightness,
            Augmentor.CONTRAST : self._contrast
        }
        self.augment_functions_single = {
            Augmentor.MIRROR_V : self._mirrorVSingle,
            Augmentor.MIRROR_H : self._mirrorHSingle,
            Augmentor.ROTATION : self._rotateSingle,
            Augmentor.ZOOM : self._zoomSingle,
            Augmentor.BRIGHTNESS : self._brightnessSingle,
            Augmentor.CONTRAST : self._contrastSingle
        }
        self.rng_seed = rng
        self._defineGenerator()
        self.aug_chance = aug_chance if aug_chance is not None else Augmentor.DEFAULT_AUGMENTATION_CHANCE

    def _defineGenerator(self):
        """Defines the random number generator from the seed passed in the constructor."""
        self.generator = self.rng_seed if isinstance(self.rng_seed, np.random.RandomState) else np.random.RandomState(self.rng_seed)
    
    def resetRandomState(self) -> None:
        """Resets the random state of the augmentor. This works only if the constructor received an integer seed."""
        self._defineGenerator()

    def augment(self, augmentations : Sequence[int], imageset : Sequence[Union[np.ndarray, Sequence[np.ndarray]]], is_mask : Sequence[bool], report : bool = False) -> List[List[np.ndarray]]:
        """
        Augments an entire dataset - creates a new dataset by running the specified augmentations.
        This function shouldn't be called on batches during training.

        Arguments:
        - 'augmentations' - A list of augmentations which will be applied to the dataset.
        - 'imageset' - The augmented dataset - It should be a sequence of data arrays, e.g., [array_of_images, array_of_masks, array_of_labels] if each sample has an image, mask and a label.
        - 'is_mask' - Which of the data arrays in 'imageset' represent masks and shouldn't receive brightness/contrast augmentations.
        - 'report' - Whether to print out the progress of the augmentation.

        Returns:
        - A list of data lists of augmented samples, e.g., [list_of_augmented_images, list_of_augmented_masks, list_of_augmented_labels] if a sample has an image, mask and a label.
        """
        if len(imageset) == 0:
            raise ValueError("No dataset to augment.")
        start = time.time()
        num_samples = len(imageset[0])
        augmented_set = [[] for _ in range(len(imageset))]
        for sample_idx in range(num_samples):
            for aug_code in augmentations:
                augmented_sample = self.augment_functions[aug_code]([single_set[sample_idx] for single_set in imageset], is_mask)
                for img_idx in range(len(augmented_sample)):
                    for augmented_sample_idx in range(len(augmented_sample[img_idx])):
                        augmented_set[img_idx].append(augmented_sample[img_idx][augmented_sample_idx])
            if report:
                print("Processed sample {}/{}".format(sample_idx + 1, num_samples), end="\r")
        if report:
            print()
            print("Augmentation finished in {:.2f} seconds.".format(time.time() - start))
        return augmented_set
    
    def augmentSampleMulti(self, augmentations : Sequence[int], sample : Sequence[np.ndarray], is_mask : Sequence[bool], report : bool = False) -> List[np.ndarray]:
        """
        Augments a single data sample with multiple augmentation at once. It uses 'aug_chance' to determine whether the sample should be augmented by each one.

        Arguments:
        - 'augmentations' - Sequence of augmentations, which should applied to the sample with probability given by 'aug_chance' in constructor.
        - 'sample' - Sequence of image-like numpy arrays representing the sample.
        - 'is_mask' - Sequence of bools determining, which images shouldn't have their brightness/contrast augmented.
        - 'report' - Unused.

        Returns:
        - Augmented sample.
        """
        augmented_sample = sample
        for aug_code in augmentations:
            rnd_chance = self.generator.random()
            if rnd_chance < self.aug_chance:
                augmented_sample = self.augment_functions_single[aug_code](augmented_sample, is_mask)
        return augmented_sample
    
    def augmentSamplesMulti(self, augmentations : Sequence[int], samples : Sequence[Sequence[np.ndarray]], is_mask : Sequence[bool], report : bool = False) -> List[List[np.ndarray]]:
        """
        Augments multiple samples with multiple augmentations at once with the probability 'aug_chance' given in constructor.
        This method should be applied to bathces during training.

        Arguments:
        - 'augmentations' - Sequence of augmentations, which should applied to the samples with probability given by 'aug_chance' in constructor.
        - 'sample' - Sequence of samples.
        - 'is_mask' - Sequence of bools determining, which images in a sample shouldn't have their brightness/contrast augmented.
        - 'report' - Unused.

        Returns:
        - List of augmented samples.
        """
        augmented_samples = []
        for sample in samples:
            augmented_samples.append(self.augmentSampleMulti(augmentations, sample, is_mask, report))
        return augmented_samples

    def _mirrorV(self, sample : Sequence[np.ndarray], is_mask : Sequence[bool]) -> List[List[np.ndarray]]:
        """Random vertical flip/mirror. The result will be a list of lists of images."""
        return [[np.flipud(img)] for img in sample]
    
    def _mirrorVSingle(self, sample : Sequence[np.ndarray], is_mask : Sequence[bool]) -> List[np.ndarray]:
        """Random vertical flip/mirror. The result will be a list of images."""
        return [np.flipud(img) for img in sample]

    def _mirrorH(self, sample : Sequence[np.ndarray], is_mask : Sequence[bool]) -> List[List[np.ndarray]]:
        """Random horizontal flip/mirror. The result will be a list of lists of images."""
        return [[np.fliplr(img)] for img in sample]
    
    def _mirrorHSingle(self, sample : Sequence[np.ndarray], is_mask : Sequence[bool]) -> List[np.ndarray]:
        """Random horizontal flip/mirror. The result will be a list of images."""
        return [np.fliplr(img) for img in sample]

    def _rotate(self, sample : Sequence[np.ndarray], is_mask : Sequence[bool]) -> List[List[np.ndarray]]:
        """Random rotation. The result will be a list of lists of images. Applies 'rot_count' augmentations."""
        result = [[] for _ in range(len(sample))]
        for _ in range(self.rot_count):
            rnd_angle = self.generator.random() * (self.rot_bounds[1] - self.rot_bounds[0]) + self.rot_bounds[0]
            for img_idx in range(len(sample)):
                result[img_idx].append(skimage.transform.rotate(sample[img_idx], rnd_angle, mode="reflect"))
        return result
    
    def _rotateSingle(self, sample : Sequence[np.ndarray], is_mask : Sequence[bool]) -> List[np.ndarray]:
        """Random rotation. The result will be a list of images."""
        result = []
        rnd_angle = self.generator.random() * (self.rot_bounds[1] - self.rot_bounds[0]) + self.rot_bounds[0]
        for img in sample:
            result.append(skimage.transform.rotate(img, rnd_angle, mode="reflect"))
        return result

    def _zoom(self, sample : Sequence[np.ndarray], is_mask : Sequence[bool]) -> List[List[np.ndarray]]:
        """Random zoom. The result will be a list of lists of images. Applies 'zoom_count' augmentations."""
        result = [[] for _ in range(len(sample))]
        for _ in range(self.zoom_count):
            zoom_fac = (self.generator.random() * (self.zoom_bounds[1] - self.zoom_bounds[0])) + self.zoom_bounds[0]
            pad_fac = (1 / zoom_fac) - 1
            for img_idx in range(len(sample)):
                if zoom_fac < 1:
                    pad_width = (int(pad_fac * 0.5 * sample[img_idx].shape[0]), int(pad_fac * 0.5 * sample[img_idx].shape[1]))
                    pad_width = (pad_width, pad_width) if sample[img_idx].ndim == 2 else (pad_width, pad_width, (0, 0))
                    result[img_idx].append(skimage.transform.resize(np.pad(sample[img_idx], pad_width=pad_width, mode="reflect"), sample[img_idx].shape[:2]))
                else:
                    output_shape = (zoom_fac * sample[img_idx].shape[0], zoom_fac * sample[img_idx].shape[1])
                    corner = (int((output_shape[0] - sample[img_idx].shape[0]) / 2), int((output_shape[1] - sample[img_idx].shape[1]) / 2))
                    result[img_idx].append(skimage.transform.resize(sample[img_idx], output_shape=output_shape)[corner[0] : corner[0] + sample[img_idx].shape[0], corner[1] : corner[1] + sample[img_idx].shape[1]])
        return result
    
    def _zoomSingle(self, sample : Sequence[np.ndarray], is_mask : Sequence[bool]) -> List[np.ndarray]:
        """Random zoom. The result will be a list of images."""
        result = []
        zoom_fac = (self.generator.random() * (self.zoom_bounds[1] - self.zoom_bounds[0])) + self.zoom_bounds[0]
        pad_fac = (1 / zoom_fac) - 1
        for img in sample:
            if zoom_fac < 1:
                pad_width = (int(pad_fac * 0.5 * img.shape[0]), int(pad_fac * 0.5 * img.shape[1]))
                pad_width = (pad_width, pad_width) if img.ndim == 2 else (pad_width, pad_width, (0, 0))
                result.append(skimage.transform.resize(np.pad(img, pad_width=pad_width, mode="reflect"), img.shape[:2]))
            else:
                output_shape = (zoom_fac * img.shape[0], zoom_fac * img.shape[1])
                corner = (int((output_shape[0] - img.shape[0]) / 2), int((output_shape[1] - img.shape[1]) / 2))
                result.append(skimage.transform.resize(img, output_shape=output_shape)[corner[0] : corner[0] + img.shape[0], corner[1] : corner[1] + img.shape[1]])
        return result

    def _brightness(self, sample : Sequence[np.ndarray], is_mask : Sequence[bool]) -> List[List[np.ndarray]]:
        """Random brightness. The result will be a list of lists of images. Applies 'brightness_count' augmentations."""
        result = [[] for _ in range(len(sample))]
        for _ in range(self.brightness_count):
            intensity_fac = self.generator.random() * (self.brightness_range[1] - self.brightness_range[0]) + self.brightness_range[0]
            for img_idx in range(len(sample)):
                result[img_idx].append(sample[img_idx] if is_mask[img_idx] else np.clip(sample[img_idx] * intensity_fac, 0, 1))
        return result
    
    def _brightnessSingle(self, sample : Sequence[np.ndarray], is_mask : Sequence[bool]) -> List[np.ndarray]:
        """Random brightness. The result will be a list of images."""
        result = []
        intensity_fac = self.generator.random() * (self.brightness_range[1] - self.brightness_range[0]) + self.brightness_range[0]
        for img_idx in range(len(sample)):
            result.append(sample[img_idx] if is_mask[img_idx] else np.clip(sample[img_idx] * intensity_fac, 0, 1))
        return result

    def _contrastChange(self, img : np.ndarray, mul : float) -> np.ndarray:
        """
        Applies non-linear contrast modification to the image.
        The single parameter of this function defiens the exponent in contrast modification.
        - Negative values mean reduced contrast and positive numebr means increased contrast.

        Arguments:
        - 'img' - The modified image.
        - 'mul' - The parameter defining the exponent in contrast modifying algorithm.

        Returns:
        - Image with updated contrast.
        """
        mul_sign = mul > 0
        mul_val = np.abs(mul) + 1.0

        mean_val = np.mean(img)
        max_val = np.max(img)
        min_val = np.min(img)
        mask = img > mean_val
        
        if mul_sign:
            less_vals = (img - min_val) / (mean_val - min_val)
            more_vals = (max_val - img) / (max_val - mean_val)
        else:
            less_vals = (mean_val - img) / (mean_val - min_val)
            more_vals = (img - mean_val) / (max_val - mean_val)
        
        less_vals = np.abs(less_vals) ** mul_val
        more_vals = np.abs(more_vals) ** mul_val

        if mul_sign:
            less_vals = less_vals * (mean_val - min_val) + min_val
            more_vals = max_val - more_vals * (max_val - mean_val)
        else:
            less_vals = mean_val - less_vals * (mean_val - min_val)
            more_vals = more_vals * (max_val - mean_val) + mean_val

        result = np.zeros_like(img)
        result[np.logical_not(mask)] = less_vals[np.logical_not(mask)]
        result[mask] = more_vals[mask]
        
        return result

    def _contrast(self, sample : Sequence[np.ndarray], is_mask : Sequence[bool]) -> List[List[np.ndarray]]:
        """Random contrast. The result will be a list of lists of images. Applies 'contrast_count' augmentations."""
        result = [[] for _ in range(len(sample))]
        for _ in range(self.contrast_count):
            contrast_fac = self.generator.random() * (self.contrast_range[1] - self.contrast_range[0]) + self.contrast_range[0]
            for img_idx in range(len(sample)):
                result[img_idx].append(sample[img_idx] if is_mask[img_idx] else self._contrastChange(sample[img_idx], contrast_fac))
        return result
    
    def _contrastSingle(self, sample : Sequence[np.ndarray], is_mask : Sequence[bool]) -> List[np.ndarray]:
        """Random contrast. The result will be a list of images."""
        result = []
        contrast_fac = self.generator.random() * (self.contrast_range[1] - self.contrast_range[0]) + self.contrast_range[0]
        for img_idx in range(len(sample)):
            result.append(sample[img_idx] if is_mask[img_idx] else self._contrastChange(sample[img_idx], contrast_fac))
        return result
