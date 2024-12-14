
import time
from pathlib import Path
from typing import List, Dict

import numpy as np

from .general import RFI, RFISet

class DatasetLoader:
    """
    Base loader class with common methods.
    """

    _GROUP_ALL_NAME = "all"
    _DEFAULT_NAME = "'UNKNOWN DATASET'"

    def __init__(self, root_path : Path) -> None:
        """
        Initialises the Dataset loading construct. It calls the '_create' function
        which goes through the folder directory to prepare descriptions of data samples.

        Arguments:
        - 'root_path' - Path to the root of the dataset file hierarchy as downloaded.
        """
        self.root_path = root_path
        self.descriptions : List[Dict[str, object]] = None
        self.samples : List[RFI] = None
        self.group_indices : Dict[str, np.ndarray] = None
        self._create()

    def _create(self) -> None:
        """Unimplemented dataset description creation function."""
        raise NotImplementedError()
    
    def load(self, idx : int) -> None:
        """Unimplemented single sample loading function."""
        raise NotImplementedError()
    
    def loadAll(self) -> None:
        """
        Loads all samples from the dataset. This may take non-trivial time becasue it has to load
        many images and masks.
        """
        for idx in range(len(self.descriptions)):
            self.load(idx)

    def __getitem__(self, idx : int) -> RFI:
        """
        Returns the 'RFI' sample at the given index. It loads the sample if necessary.

        Arguments:
        - 'idx' - Index of the requested sample.

        Returns:
        - The 'RFI' sample at the requested index.
        """
        return self.get(idx = idx)
    
    def __len__(self) -> int:
        """
        Returns the number of samples in this dataset. It will be the sum of the sizes of train and test sets.

        Returns
        - The number of samples in the entire dataset.
        """
        return len(self.descriptions)

    def get(self, idx : int = None, name : str = None, **kwargs) -> RFI:
        """
        Returns a sample from the dataset. If the sample hasn't been loaded so far, the function
        loads and then returns. The samples are cached in the object and will not be loaded multiple
        times. The sample can be requested according to its index or name. One can also specify
        'set' argument to request indexing into a specific subset of the dataset.
        At least one of 'idx' and 'name' has to be specified and if both are given then a sample
        according to 'name' is returned.

        Arguments:
        - 'idx' - Index of the requested sample.
        - 'name' - Name of the requested sample. It is the stem of the file name.
        - 'set' - Name of the set where the sample should be taken from.

        Returns:
        - 'RFI' object containing the loaded images and masks.
        """
        if name is not None:
            nameIdx = -1
            for i, desc in enumerate(self.descriptions):
                if name == desc["stem"]:
                    nameIdx = i
                    break
            if nameIdx < 0:
                raise ValueError("Specified name: '{}' was not found in the ChaseDB1 dataset.".format(name))
            else:
                self.load(nameIdx)
                return self.samples[nameIdx]
        if idx is None:
            raise ValueError("Either 'name' or 'idx' has to be specified.")
        if "set" in kwargs.keys():
            if kwargs["set"] not in self.group_indices.keys():
                raise ValueError("Unknown ChaseDB1 set: '{}'.".format(kwargs["set"]))
            self.load(self.group_indices[kwargs["set"]][idx])
            return self.samples[self.group_indices[kwargs["set"]][idx]]
        self.load(idx)
        return self.samples[idx]
    
    def getDatasetName(self) -> str:
        return DatasetLoader._DEFAULT_NAME
    
    def getSet(self, name : str = None, indices : np.ndarray = None, new_name : str = None) -> RFISet:
        """
        Returns a complete set of samples according to the name 'name' of a subset.
        The samples from the requested set will be loaded before the set is returned,
        which can take non-trivial time. This function should be called only if
        you are certain that you will use the whole set and not just some samples from it.

        Alternatively the function can return a set of samples composed from the selected indices 'indices'.
        This can be used to define a custom set using indices. If both name and indices are specified
        then the function creates a subset of the subset according to indices.

        Arguments:
        - 'name' - Name of the selected FIVES subset. All by default.
        - 'indices' - Selected indices from the given subset. All by default.
        - 'new_name' - New name assigned to the returned set of RFIs.

        Returns:
        - A 'RFISet' object with selected samples from the dataset.
        """
        complete_indices = self.group_indices[DatasetLoader._GROUP_ALL_NAME]
        if name is not None:
            if name not in self.group_indices.keys():
                raise ValueError("The requested set: {} does not exist in {} hierarchy.".format(name, self.getDatasetName()))
            complete_indices = self.group_indices[name]
        subset_indices = np.arange(complete_indices.size) if indices is None else indices
        new_group_indices = complete_indices[subset_indices]
        max_name_length = np.max([len(desc["stem"]) for desc in self.descriptions])
        for i, idx in enumerate(new_group_indices):
            start_load = time.time()
            self.load(idx)
            delta_load = time.time() - start_load
            print("Loading image: {:>3}/{:>3} {:>{name_length}} in {:>5.2f} seconds".format(i + 1, new_group_indices.size, "({})".format(self.samples[idx].name), delta_load, name_length=max_name_length + 2), end="\r")
        print()
        rfi_set = RFISet(new_name, np.asarray(self.samples, object)[new_group_indices])
        return rfi_set
