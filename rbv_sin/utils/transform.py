
from typing import Tuple, Union, Callable, Sequence
import numpy as np
import skimage.transform

from rbv_sin.data import RFI, RFISet

class RFITransform:
    """
    Base class for transformation of retinal fundus image data. These trasnformations are inteded to convert
    a dataset into a form usable by further steps in the processing pipeline.
    """

    def apply(self, data : Union[RFI, RFISet]) -> Union[RFI, RFISet]:
        """
        Applies the transsformation to retinal fundus images. It accepts both a single 'RFI' sample as
        well as 'RFISet' collection, where it trasnforms each sample in the coollection.
        This function may be overriden if a more specific implementation is needed.

        Arguments:
        - 'data' - Retinal fundus data. It should be either 'RFI' or 'RFISet' object.

        Returns:
        - Returns the same data type as the input ('RFI', 'RFISet') with trasnformation applied on each sample.
        """
        return self._apply(data)
    
    def _apply(self, data : Union[RFI, RFISet]) -> Union[RFI, RFISet]:
        """Private implementation of transformation application 'Transform.apply'"""
        if isinstance(data, RFI):
            return self._applySingle(data)
        elif isinstance(data, RFISet):
            return RFISet(data.name, [self._applySingle(rfi) for rfi in data])
        else:
            raise TypeError("The data type in transformation is neither 'RFI' nor 'RFISet'.")
        
    def _applySingle(self, rfi : RFI) -> RFI:
        """
        This function should be overriden with a concrete trasnformation routine for a single 'RFI' sample.
        The implementation should use 'self._solveRFI' to fit into the API.
        """
        raise NotImplementedError()

    def _solveList(self, data : Union[Sequence[np.ndarray], np.ndarray, None], func : Callable[[np.ndarray], np.ndarray]) -> Union[Sequence[np.ndarray], np.ndarray, None]:
        """
        Applies the given function to either a single numpy array, each item from a list of numpy arrays
        or it returns None if the input 'data' is None.

        Arguments:
        - 'data' - Trasnformed data: either numpy ndarray, sequence of numpy ndarrays or None.
        - 'func' - Function, which transforms the numpy arrays - it should carry necessary parameters in its scope.

        Returns:
        - The same as 'data' after transformation of each item by 'func'.
        """
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return func(data)
        return [func(item) for item in data]
    
    def _solveRFI(self, rfi : RFI, func : Callable[[np.ndarray], np.ndarray]) -> RFI:
        """
        Applies the given function to all elements of a 'RFI' sample. The elements can be None,
        numpy arrays or lists of numpy arrays. Returns a 'RFI sample' with the same attributes
        as on the input after trasnformation by 'func'.

        Arguments:
        - 'rfi' - A 'RFI' sample whose attributes need to be trasnformed.
        - 'func' - The transformation routine for a single numpy array.

        Returns:
        - A 'RFI' sample with transformed attributes.
        """
        new_rfi = RFI(rfi.name)
        new_rfi.image = self._solveList(rfi.image, func)
        new_rfi.mean_od_mask = self._solveList(rfi.mean_od_mask, func)
        new_rfi.mean_oc_mask = self._solveList(rfi.mean_oc_mask, func)
        new_rfi.od_mask = self._solveList(rfi.od_mask, func)
        new_rfi.oc_mask = self._solveList(rfi.oc_mask, func)
        new_rfi.vessel_mask = self._solveList(rfi.vessel_mask, func)
        return new_rfi

class RFIResize(RFITransform):
    """
    Scales all rfi samples to the given target shape. Retinal images are generally quite large and we want
    to scale them down. On top of that, neural networks mostly require a fixed scale of the input and this
    class can be used to achieve that.
    """

    def __init__(self, target_shape : Tuple[int]) -> None:
        """
        Stores the requested shape of the resized images.

        Arguments:
        - 'target_shape' - The target shape of the resized images.
        """
        self.target_shape = target_shape

    def _applySingle(self, rfi : RFI) -> RFI:
        """
        Resizes a single 'RFI' sample by using generic 'RFITransform' methods and lambda function.
        """
        return self._solveRFI(rfi, lambda x: skimage.transform.resize(x, self.target_shape))

class RFICrop(RFITransform):
    """
    This transformation crops the retinal fundus images according to a static crop boundaries.
    """

    def __init__(self, boundaries : Tuple[int]) -> None:
        """
        Stores the static boundaries for cropping.

        Arguments:
        - 'boundaries' - Tuple of AABB indices for cropping images as returned by 'maskBoundaries'
        """
        super().__init__()
        self.boundaries = boundaries

    def _applySingle(self, rfi : RFI) -> RFI:
        """
        Crops a single 'RFI' sample according to the static boundaries stored in the constructor.
        """
        min_r, max_r, min_c, max_c, _, _ = self.boundaries
        return self._solveRFI(rfi, lambda x: x[min_r : max_r, min_c : max_c])
