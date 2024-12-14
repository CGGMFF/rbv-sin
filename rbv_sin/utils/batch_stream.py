
import numpy as np

class BatchStream:
    """The base class for batch streamers."""

    def __init__(self, single_pass : bool = True, fill_end_batches : bool = False) -> None:
        """
        Initialises the batch streamer.

        Arguments:
        - 'single_pass' - Whether the streamer should pass over the data only once.
        - 'fill_end_batches' - Whether the last batch should be filled from the beginning of the dataset if there is not enough data to create a full batch.
        """
        self.single_pass = single_pass
        self.fill_end_batches = fill_end_batches
        self.current_index = 0

    def nextBatch(self, batch_size : int):
        """Virtual method for generating the next batch."""
        raise NotImplementedError()

    def getNumSamples(self):
        """Virtual method for getting the number of samples."""
        raise NotImplementedError()

    def getNumBatches(self):
        """Virtual method for getting the number of batches."""
        raise NotImplementedError()
    
    def reset(self):
        """Resets the streamer."""
        self.current_index = 0

class ArrayStream(BatchStream):
    """Implementation of sequential batch streamer from a data array."""

    def __init__(self, *array_args, single_pass : bool = True, fill_end_batches : bool = False) -> None:
        """
        Initialisation of the array streamer with data and parameters.

        Arguments:
        - '*array_args' - Data arrays which should be sampled into batches, e.g. images and masks.
        - 'single_pass' - Whether the streamer should pass over the data arrays exactly once.
        - 'fill_end_batches' - Whether the last batch should be filled with samples from the beginning of the dataset.
        """
        super().__init__(single_pass=single_pass, fill_end_batches=fill_end_batches)
        self.data = [arr for arr in array_args]
        self.data_size = self.data[0].shape[0]

    def nextBatch(self, batch_size: int):
        """
        Samples the next batch of the given size from the arrays passed to the streamer.
        The batch will be sampled sequentially until the end of the dataset where it start looping
        if 'single_pass' was set to False.

        Arguments:
        - 'batch_size' - Size of the sampled batch.
        """
        if self.single_pass and self.current_index >= self.data_size:
            return None
        cutout = min(batch_size, self.data_size - self.current_index)
        batch = [data_array[self.current_index : self.current_index + cutout] for data_array in self.data]
        self.current_index += batch_size

        remain = 0
        if self.fill_end_batches and (cutout < batch_size):
            remain = batch_size - cutout
            remain_batch = [data_array[: remain] for data_array in self.data]
            batch = [np.concatenate([batch_block, remain_block]) for batch_block, remain_block in zip(batch, remain_batch)]
        if (not self.single_pass) and (self.current_index >= self.data_size):
            self.current_index = remain
        return batch
