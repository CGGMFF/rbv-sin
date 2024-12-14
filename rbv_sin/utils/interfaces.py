
import numpy as np

class SINSegmentation:
    """Interface for a blood vessel segmentation algorithm."""

    def __init__(self) -> None:
        pass

    def __call__(self, image : np.ndarray) -> np.ndarray:
        """Virtual method for executing segmentation."""
        raise NotImplementedError()

class SINInpainting:
    """Interface for a blood vessel inpainting algorithm."""

    def __init__(self) -> None:
        pass

    def __call__(self, image : np.ndarray, mask : np.ndarray) -> np.ndarray:
        """Virtual method for executing inpainting."""
        raise NotImplementedError()
