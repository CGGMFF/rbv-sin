
from pathlib import Path
from typing import Sequence, Union
import numpy as np

from .general import RFI, RFISet

class RFIIO:
    """
    Facilitates the loading and storing of RFI adn RFISet objects. It uses numpy (and pickle underneath)
    to store the arrays. This means that loading and storing between different version might break and
    the stored data will need to be recomputed after moving to new library versions.
    """

    ATTR_SET_NAME = "set_name"
    ATTR_NAME = "names"
    ATTR_IMAGE = "images"
    ATTR_OD = "od_masks"
    ATTR_MEAN_OD = "mean_od_masks"
    ATTR_OC = "oc_masks"
    ATTR_MEAN_OC = "mean_oc_masks"
    ATTR_VESSEL = "vessel_masks"

    def store(self, path : Path, sample : Union[RFI, RFISet]) -> None:
        """
        Stores RFI or an RFISet in a numpy zipped archive.
        Throws an error if the type of the argument is something else.

        Arguments:
        - 'path' - Path to the npz file where the class saves the data.
        - 'sample' - A RFI or RFISet object which shoud be saved.
        """
        if isinstance(sample, RFI):
            self._store(path, [sample])
        elif isinstance(sample, RFISet):
            self._store(path, sample, sample.name)
        else:
            raise TypeError("Sample type should be 'RFI' or 'RFISet'.")
        
    def _store(self, path : Path, samples : Sequence[RFI], set_name : str = "") -> None:
        """
        Takes apart the RFI objects and stores arrays of images, masks and names separately
        so that we do not have to deal with pickling custom classes.
        It saves the attributes which might be lists as object arrays.

        Arguments:
        - 'path' - Path to the npz file where the data should be stored.
        - 'samples' - A sequence of RFI objects (It can be a RFISet object).
        - 'set_name' - Name of the set - it can be left empty if it is not needed.
        """
        names = []
        images = []
        od_masks = []
        mean_od_masks = []
        oc_masks = []
        mean_oc_masks = []
        vessel_masks = []
        for sample in samples:
            names.append(sample.name)
            images.append(sample.image)
            od_masks.append(np.asarray([sample.od_mask], float) if sample.od_mask is not None else np.asarray([None]))
            mean_od_masks.append(sample.mean_od_mask)
            oc_masks.append(np.asarray([sample.oc_mask], float) if sample.oc_mask is not None else np.asarray([None]))
            mean_oc_masks.append(sample.mean_oc_mask)
            vessel_masks.append(np.asarray([sample.vessel_mask], float) if sample.vessel_mask is not None else np.asarray([None]))
        kwds = {
            RFIIO.ATTR_SET_NAME : set_name,
            RFIIO.ATTR_NAME : names,
            RFIIO.ATTR_IMAGE : images,
            RFIIO.ATTR_OD : od_masks,
            RFIIO.ATTR_MEAN_OD : mean_od_masks,
            RFIIO.ATTR_OC : oc_masks,
            RFIIO.ATTR_MEAN_OC : mean_oc_masks,
            RFIIO.ATTR_VESSEL : vessel_masks,
        }
        np.savez_compressed(path, **kwds)

    def load(self, path : Path, reduce_single : bool = True) -> Union[RFISet, RFI]:
        """
        Loads a RFI or RFISet object from files saved using this class. It can return a single RFI object
        as 1-element RFISet or just an RFI object.

        Arguments:
        - 'path' - Path to the npz file with RFI data.
        - 'reduce_single' - Whether a dataset consisting of 1 rfi should be reduced to a single RFI object
          or kept as a RFISet of size 1.

        Returns:
        - Either RFISet or RFI stored in the given file.
        """
        kwds = np.load(path, allow_pickle=True)
        rfis = []
        for name, image, od, mean_od, oc, mean_oc, vessel in zip(kwds[RFIIO.ATTR_NAME], kwds[RFIIO.ATTR_IMAGE], kwds[RFIIO.ATTR_OD], 
                                                                 kwds[RFIIO.ATTR_MEAN_OD], kwds[RFIIO.ATTR_OC], kwds[RFIIO.ATTR_MEAN_OC], kwds[RFIIO.ATTR_VESSEL]):
            rfi = RFI(name)
            rfi.image = image
            rfi.od_mask = od[0]
            rfi.mean_od_mask = mean_od
            rfi.oc_mask = oc[0]
            rfi.mean_oc_mask = mean_oc
            rfi.vessel_mask = vessel[0]
            rfis.append(rfi)
        if len(rfis) > 1 or reduce_single == False:
            rfi_set = RFISet(kwds[RFIIO.ATTR_SET_NAME], rfis)
            return rfi_set
        else:
            return rfis[0]
