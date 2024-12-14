
from typing import Tuple, List, Union

import numpy as np
import skimage.morphology

class MaskBlobs:
    """
    Class for generation of blob-like focus masks.
    The blob mask generates random circles in a larger circle until a certain percentage is filled up.
    """

    def __init__(self, target_shape : Tuple[int, int], radius_limit : int, rng : Union[int, None, np.random.RandomState]) -> None:
        """
        Initialises the blob focus mask generator.

        Arguments:
        - 'target_shape' - The shape of the mask.
        - 'radius_limit' - The pixel radius for blob generation.
        - 'rng' - The random number generation for the generator.
        """
        self.target_shape = target_shape
        self.centre = (self.target_shape[0] // 2, self.target_shape[1] // 2)
        self.radius_limit = radius_limit
        self.generator = rng if isinstance(rng, np.random.RandomState) else np.random.RandomState(rng)

    def generate(self, num_masks : int, density : float = 0.5) -> List[np.ndarray]:
        """
        Generates a requested number of blob masks with the given density - percentage of filled up circle.

        Arguments:
        - 'num_masks' - The number of requested masks.
        - 'density' - The requested percentage of coverage in the focus circle.

        Returns:
        - A list with the masks.
        """
        masks = []
        for _ in range(num_masks):
            mask = np.zeros(self.target_shape, int)
            limit_disc = skimage.morphology.disk(self.radius_limit)
            sum_limit_disc = np.sum(limit_disc)
            coverage = 0
            while coverage < density:
                self._addCircle(mask)
                mask = np.clip(mask, 0, 1)
                coverage = np.sum(mask) / sum_limit_disc
            masks.append(mask)
        return masks

    def _addCircle(self, mask : np.ndarray) -> None:
        """Private method that adds a single random circle to the generated mask."""
        rnd_angle = self.generator.random() * 2 * np.pi
        rnd_dist = self.generator.random() * self.radius_limit
        rnd_radius = self.generator.randint(5, 20)
        disc = skimage.morphology.disk(rnd_radius)
        vec = np.asarray((np.cos(rnd_angle), np.sin(rnd_angle)))
        vec = np.asarray(vec * rnd_dist, int)
        coords = (self.centre[0] + vec[0], self.centre[1] + vec[1])
        b = [coords[0] - rnd_radius, coords[0] - rnd_radius + disc.shape[0], coords[1] - rnd_radius, coords[1] - rnd_radius + disc.shape[1]]
        mask[b[0] : b[1], b[2] : b[3]] = mask[b[0] : b[1], b[2] : b[3]] + disc
