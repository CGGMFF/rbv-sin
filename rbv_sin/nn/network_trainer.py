
from typing import Tuple, Union, Sequence, Dict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from rbv_sin.nn.train_report import TrainReport
from rbv_sin.nn.train_metrics import TrainMetrics, MetricCounter
from rbv_sin.utils import batch_stream

class NetworkTrainer:
    """Base class for custom neural network training implementations adhering to the SIN methodology."""

    def __init__(self, network_inputs : Union[Sequence[Tuple[int, ...]], Tuple[int, ...], Dict[str, Tuple[int, ...]]], metric_names : Sequence[str] = []) -> None:
        """
        Initialiases the network. Sets up the metric measuring objects and defines the model by calling '_defineModel'.

        Arguments:
        - 'network_inputs' - The input shape, a sequence of input shapes or a dictionary of input shapes, which is needed in model definnition.
        - 'metric_names' - Names of losses and metrics needed during training - they can be added later but only before calling 'fit'.
        """
        self.network_inputs = network_inputs
        self.metric_names = metric_names
        self.current_epoch = 0
        self.current_step = 0

        self.train_metrics = TrainMetrics(self.metric_names, None)
        self.epoch_metrics = MetricCounter(self.metric_names)

        self.setCheckpointParams()
        self._defineModel()

    def _defineModel(self) -> None:
        """Virtual method for model definition."""
        raise NotImplementedError("There is no model definition!")

    def compileModel(self, batch_size : int = 1, epochs : int = 1, batches_per_epoch : Union[int, None] = None, cp_epoch_delta : int = 1) -> None:
        """
        Sets up the base training parameters for the network. The model needs to be compiled only for training.

        Arguments:
        - 'batch_size' - The batch size used during training.
        - 'epochs' - Number of training epochs.
        - 'batches_per_epoch' - Number of batches sampled during training, 'None' means that the number is according to the dataset size.
        - 'cp_epoch_delta' - How often should the network save a checkpoint.
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.batches_per_epoch = batches_per_epoch
        self.cp_epoch_delta = cp_epoch_delta
        self.last_saved_epoch = None
        self.reporter = TrainReport()

    def saveMetricName(self) -> str:
        """Name of a metric, which will be written in the checkpoints folder name."""
        return None

    def _trainStep(self, batch_images : tf.Tensor, batch_labels : tf.Tensor) -> Dict[str, float]:
        """Virtual method for the implementation of one training step - with one batch."""
        raise NotImplementedError()
    
    def _transformBatch(self, batch_from_streamer : Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Virtual method for the implementation of batch transformation."""
        raise NotImplementedError()
    
    def _trainEpoch(self) -> None:
        """
        Runs the algorithm for a single epoch. This function samples batches, transforms them and computes
        the training step for each sampled batch.
        Updates the registered metrics according to the losses returned by '_trainStep'.
        """
        self.epoch_metrics.reset()
        batch_idx = 0

        self.streamer.reset()
        batch = self.streamer.nextBatch(self.batch_size)
        while batch is not None:
            self.current_step += 1
            batch_images, batch_labels = self._transformBatch(batch)

            batch_losses = self._trainStep(tf.convert_to_tensor(batch_images), tf.convert_to_tensor(batch_labels))
            self.epoch_metrics.addStep(batch_losses)
            batch_idx += 1
            self.train_metrics.step(batch_losses)
            self.reporter.updateEpoch(batch_idx, self.epoch_metrics.getActiveValues(mean=True))

            batch = self.streamer.nextBatch(self.batch_size)
            if (self.batches_per_epoch is not None) and (batch_idx >= self.batches_per_epoch):
                batch = None

    def _validateEpoch(self, validation_images, validation_labels) -> Dict[str, float]:
        """Virtual method for the implementation of epoch validation."""
        raise NotImplementedError()
    
    def fit(self, images : np.ndarray, vessels : np.ndarray, val_images : np.ndarray = None, val_labels : np.ndarray = None) -> None:
        """
        Starts the training algorithm. The network has to have been compiled before fitting.
        Validation data are optional and the model will be validated only if both validation input 'images' and labels 'labels' are provided.
        This method will call 'saveMetricName' to set the metric name before beginning the fitting.

        Arguments:
        - 'images' - Training set of images.
        - 'vessels' - Training set of blood vessels.
        - 'val_images' - A set of validation inputs or None.
        - 'val_labels' - A set of validation labels or None.
        """
        self.validation = (val_images is not None) and (val_labels is not None)
        if self.validation:
            self.train_metrics.setSaveMetric(self.saveMetricName())

        self.streamer = batch_stream.ArrayStream(images, vessels, single_pass=(self.batches_per_epoch is None), fill_end_batches=True)
        report_batches_per_epoch = self.batches_per_epoch if self.batches_per_epoch is not None else (images.shape[0] + self.batch_size - 1) // self.batch_size

        self.iteration_start_epochs = self.current_epoch
        for _ in range(self.epochs):
            self.current_epoch += 1
            self.train_metrics.startEpoch()
            self.reporter.startEpoch(self.current_epoch, self.iteration_start_epochs + self.epochs, 0, report_batches_per_epoch)
            self._trainEpoch()
            validation_result = self._validateEpoch(val_images, val_labels) if self.validation else {}
            self.reporter.endEpoch(validation_result)
            self.train_metrics.endEpoch(validation_result)
            self._onEpochEnd()

        if (self.last_saved_epoch is None) or (self.current_epoch != self.last_saved_epoch):
            self.saveCheckpoint()

    def _predictBatch(self, images : np.ndarray) -> np.ndarray:
        """Virtual method for predicting a single batch."""
        raise NotImplementedError()

    def predict(self, images : np.ndarray, batch_size : int = 1) -> np.ndarray:
        """
        Computes the mdoel prediction for a set of inputs.
        This function works only if there a single output.

        Arguments:
        - 'images' - A set of inputs for the network.
        - 'batch_size' - The predicted set will be split into batches of this size.

        Returns:
        - Returns the network results.
        """
        batch_indices = np.arange(batch_size, images.shape[0], batch_size)
        split_data = np.split(images, batch_indices)

        results = []
        for batch in split_data:
            result = self._predictBatch(batch)
            results.append(result)
        results = np.concatenate(results)

        return results
    
    def _shouldSave(self) -> bool:
        """Returns a checkpoint should be saved at the end of this epoch."""
        return (self.current_epoch > 0) and (self.current_epoch % self.cp_epoch_delta == 0)

    def _onEpochEnd(self):
        """Epoch finalisation. By default, it saves a checkpoint."""
        if self._shouldSave():
            self.saveCheckpoint()

    def setCheckpointParams(self, cp_path_train_dir : Path = None, cp_name : str = None) -> None:
        """Saves the path to the network checkpoint directory."""
        self.cp_path_train_dir = cp_path_train_dir
        self.cp_name = cp_name

    def _saveModels(self, model_dir_path : Path) -> None:
        """Virtual method for saving of model weights."""
        raise NotImplementedError()
    
    def _loadModels(self, model_dir_path : Path) -> None:
        """Virtual method for model weight loading."""
        raise NotImplementedError()
    
    def _saveExamples(self, cp_dir : Path) -> None:
        """Base empty method for saving (validation) result examples."""
        return None
    
    def _saveGraphs(self, cp_dir : Path) -> None:
        """Base empty method for saving training graphs."""
        return None

    def saveCheckpoint(self) -> None:
        """
        Saves the checkpoint of the model according to the checpoint parmeters set in 'setCheckpointParams'.
        This method saves the network parameters, then it calls the '_saveModels' method.
        After, it calls '_saveExamples', then it saves the model metrics and ends with calling '_saveGraphs'.
        """
        self.last_saved_epoch = self.current_epoch
        target_dir = Path(self.cp_path_train_dir, self.train_metrics.saveName(self.current_epoch, self.cp_name))
        Path.mkdir(target_dir, exist_ok=True)

        model_dir = Path(target_dir, "model")
        Path.mkdir(model_dir, exist_ok=True)
        with open(Path(model_dir, "model_params.txt"), "w") as params_file:
            params_file.write("{}\n".format(self.current_epoch))
            params_file.write("{}\n".format(self.current_step))
        self._saveModels(model_dir)

        self._saveExamples(target_dir)
        self.train_metrics.saveMetrics(target_dir)
        self._saveGraphs(target_dir)

    def loadCheckpoint(self, target_dir : Path) -> None:
        """
        Loads the given checkpoint by, first, loading the model weights through '_loadModels', then it loads
        the network parameters and metrics. 
        """
        model_dir = Path(target_dir, "model")
        self._loadModels(model_dir)
        with open(Path(model_dir, "model_params.txt"), "r") as params_file:
            self.current_epoch = int(params_file.readline())
            self.current_step = int(params_file.readline())

        self.train_metrics.loadMetrics(target_dir)

    def numberOfParameters(self) -> Dict[str, int]:
        """Virtual method, which should compute the number of parameters."""
        raise NotImplementedError()

    def _plotMetrics(self, collection_name, metric_names, title, labels, colors, file_path, y_ticklabels = None, y_lim = None, fig_size : Tuple[int, int] = (9, 7), show : bool = True) -> bool:
        """
        Plots and saves a single graph with several metrics.

        Arguments:
        - 'collection_name' - Name of the collection from train metrics counter, by default either 'epoch' or 'step'.
        - 'metric_names' - List of metrics that should be plotted in the graph.
        - 'title' - Title of the graph.
        - 'labels' - Labels of the metrics.
        - 'colors' - Colours of the metric plots.
        - 'file_path' - The path to the file where the graph will be saved.
        - 'y_ticklabels' - Labels of the y axis ticks.
        - 'y_lim' - The limits of the y axis.
        - 'fig_size' - The size of the figure in inches.
        - 'show' - Whether the generated graph should be displayed on top of being saved to disk.

        Returns:
        - Whether the graph was created successfully, i.e. there was at least one metric which could be drawn.
        """
        filtered_values = []
        filtered_labels = []
        filtered_colors = []
        for idx, metric_name in enumerate(metric_names):
            if self.train_metrics.isActive(collection_name, metric_name):
                filtered_values.append(self.train_metrics.getValues(collection_name, metric_name))
                filtered_labels.append(labels[idx])
                filtered_colors.append(colors[idx])
        if len(filtered_values) == 0:
            return False
        x_values = np.arange(len(filtered_values[0]))
        
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.set_title(title)
        if y_lim is not None:
            ax.set_ylim(y_lim[0], y_lim[1])
        if y_ticklabels is not None:
            ax.set_yticks(y_ticklabels)
        for (y_values, label, color) in zip(filtered_values, filtered_labels, filtered_colors):
            ax.plot(x_values, y_values, marker="o", c=color, label=label)
        ax.legend()
        fig.tight_layout()
        fig.savefig(file_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return True
