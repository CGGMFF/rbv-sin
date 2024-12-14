
from typing import Tuple, Dict
from pathlib import Path
import numpy as np

from rbv_sin.utils import interfaces
from rbv_sin.utils.inp_data_gen import InpaintDataTransformer
from rbv_sin.utils.image_splitter import ImageSplitter
from rbv_sin.nn import seg_trainer, inp_trainer
    
class SINSegmentationTileWrapper(interfaces.SINSegmentation):
    """Wrapper for our segmentation algorithm on tiled images."""

    def __init__(self, cp_path : Path, input_shape : Tuple[int, ...], splitter : ImageSplitter) -> None:
        """
        Initialises our segmentation algorithm for tiled images.

        Arguments:
        - 'cp_path' - Checkpoint directory.
        - 'input_shape' - The network input shape.
        - 'splitter' - The image splitter for tiling and image composition.
        """
        super().__init__()
        self.cp_path = cp_path
        self.input_shape = input_shape
        self.splitter = splitter
        self.network = seg_trainer.VesselSegmentationTrainer(self.input_shape)
        self.network.loadCheckpoint(self.cp_path)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Applies our segmentation algorithm by splitting the input image into patches using the 'splitter'.
        Then the patches are segmented using the network and the result is composed together using the 'splitter'.

        Arguments:
        - 'image' - The image for blood vessel segmentation.

        Returns:
        - The composed vessel segmentation mask.
        """
        image_patches = self.splitter.split(image)
        if image_patches.ndim < 4:
            image_patches = np.expand_dims(image_patches, 0)
        prediction_patches = np.squeeze(self.network.predict(image_patches))
        prediction = self.splitter.compose(prediction_patches, image.shape[:2])
        return np.squeeze(prediction)

class SINInpaintingWrapper(interfaces.SINInpainting):
    """Wrapper for our inpainting algorithm."""

    def __init__(self, cp_path : Path, input_shapes : Dict[str, Tuple[int, ...]], data_transformer : InpaintDataTransformer) -> None:
        """
        Initialises our inpainting algorithm.

        Arguments:
        - 'cp_path' - Checkpoint directory.
        - 'input_shapes' - The dictionary of the inpainting network inputs.
        - 'data_transformer' - The data transformer applied on the input image and mask.
        """
        super().__init__()
        self.cp_path = cp_path
        self.input_shapes = input_shapes
        self.data_transformer = data_transformer
        self.network = inp_trainer.VesselInpaintingTrainer(self.input_shapes)
        self.network.loadCheckpoint(self.cp_path)

    def __call__(self, image : np.ndarray, mask : np.ndarray) -> np.ndarray:
        """
        Applies our inpainting algorithm by trasnforming the image and mask into the network input and predicting
        the inpainted result.

        Arguments:
        - 'image' - The image to be inpainted.
        - 'mask' - The mask to be inpainted.

        Returns:
        - The inpainted image.
        """
        inp_input = self.data_transformer.transform(image, mask)
        if inp_input.ndim < 4:
            inp_input = np.expand_dims(inp_input, 0)
        inpainted = self.network.predict(inp_input)
        return np.squeeze(inpainted)
