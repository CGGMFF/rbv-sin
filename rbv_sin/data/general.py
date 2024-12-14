
from __future__ import annotations
from typing import Sequence, Union, List
import numpy as np

class RFI:
    """
    RFI is an abbreviation of Retinal Fundus Image and this class contains the general data items
    describing images and their ground truth from a dataset. The general nature doesn't allow
    for specific dataset-related attributes but ensures that image processing algorithms can
    rely on a simple structure consisting of the most important RFI parameters.
    """

    def __init__(self, name : Union[str, None] = None) -> None:
        """
        Initialises a RFI item with an optional name by which it can be referred to at a later
        point in time.

        Arguments:
        - 'name' - Optional name of this RFI item.
        """
        self.name : Union[str, None] = name
        self.image : Union[np.ndarray, None] = None
        self.od_mask : Union[Sequence[np.ndarray], np.ndarray, None] = None
        self.mean_od_mask : Union[np.ndarray, None] = None
        self.oc_mask : Union[Sequence[np.ndarray], np.ndarray, None] = None
        self.mean_oc_mask : Union[np.ndarray, None] = None
        self.vessel_mask : Union[Sequence[np.ndarray], np.ndarray, None] = None

class RFISet:
    """
    RFISet is a collection of Retinal Fundus Images with their ground truth segmentation masks.
    This class contains an arbitrary number of 'RFI' objects, which should be treated as a single
    dataset. It can be used, for instance, to split a larger dataset into training, validation
    and testing dataset for training and evaluation of machine learning algorithms.
    It also specifies the name of the dataset, which can be referred to during dataset processing.
    """

    def __init__(self, name : Union[str, None] = None, dataset : Union[Sequence[RFI], None] = None) -> None:
        """
        Initialises the dataset with an optional descriptive name and a sequence of 'RFI' objects.

        Arguments:
        - 'name' - Optional name of the dataset for future reference.
        - 'dataset' - A sequence of objects composing the dataset.
        """
        self.name : Union[str, None] = name
        self.dataset : Union[List[RFI], None] = dataset

    def fill(self, name : Union[str, None], dataset : Sequence[RFI]) -> None:
        """
        Fills the attributes of the dataset.

        Arguments:
        - 'name' - An optional name of the dataset for future reference.
        - 'dataset' - The set of 'RFI' objects composing the dataset.
        """
        self.name = name
        self.dataset = list(dataset)

    def add(self, rfi : Union[RFI, RFISet, Sequence[RFI]]) -> None:
        """
        Adds a new RFI object at the end of this dataset or extends the RFISet by a second RFISet
        or a sequence of RFI objects. It doesn't change the name of this RFISet.

        Arguments:
        - 'rfi' - RFI object/RFISet or a sequence of RFI objects to add to the set.
        """
        if self.dataset is None:
            self.dataset = []
        if isinstance(rfi, RFI):
            self.dataset.append(rfi)
        else:
            self.dataset.extend(rfi)

    def subset(self, indices : np.ndarray, name : Union[str, None] = None) -> RFISet:
        """
        Returns a new instance of RFISet, which will contain a subset of this instance given
        by the provided indices. The new subset instance will have the name given in the arguments.
        The content (dataset items) are not copied, the subset will reference the same objects
        as the original dataset.

        Arguments:
        - 'indices' - The indices used for the subset selection. They have to be from the range 0..len(instance).
        - 'name' - Name of the created subset.

        Return:
        - A new instance of RFISet containing a subset of the data items according to the given indices.
        """
        return RFISet(name, list(np.asarray(self.dataset, object)[indices]))

    def __iter__(self) -> RFISet:
        """
        Starts iteration over the set.

        Returns:
        - 'self' as the iterator is the object itself.
        """
        self.current = 0
        return self

    def __next__(self) -> RFI:
        """
        Returns the next item in the set during iteration.
        
        Returns:
        - 'RFI' object the current iterator position.
        """
        if self.current < len(self.dataset):
            sample = self.dataset[self.current]
            self.current += 1
            return sample
        else:
            raise StopIteration()
        
    def __getitem__(self, idx : int) -> RFI:
        """
        Returns the 'RFI' sample at the given index.

        Arguments:
        - 'idx' - Index of the requested sample.

        Returns:
        - The 'RFI' sample at the requested index.
        """
        return self.dataset[idx]
    
    def __len__(self) -> int:
        """
        Returns the number of samples in this set.

        Returns
        - The number of samples in this set.
        """
        return len(self.dataset)
    
    def join(name : Union[str, None] = None, datasets : Sequence[Union[RFISet, Sequence[RFI]]] = []) -> RFISet:
        """
        Concatenates the given sets of RFIs (eiter a sequence or RFISet) into a single RFISet
        with the given name.

        Arguments:
        - 'name' - Name of the concatenated set.
        - 'datasets' - A sequence of RFI sequences or RFISet objects which will be concatenated into a single RFISet object.

        Returns:
        - A RFISet created as a concatenation of the sets given in arguments.
        """
        rfiset = RFISet(name)
        for dataset in datasets:
            rfiset.add(dataset)
        return rfiset
