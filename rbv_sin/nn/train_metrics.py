
from datetime import datetime
from typing import List, Dict, Sequence, Union
from pathlib import Path
import numpy as np

class TrainMetrics:
    """
    Class for managing losses and metrics during network.
    This class together with 'MetricSummer' and 'TrainReport' have some overlap but they serve different purpose.
    """

    EPOCH_NAME = "epoch"
    STEP_NAME = "step"

    def __init__(self, metric_names : Sequence[str] = [], save_metric : str = None) -> None:
        """
        Initialises the metric epoch and step lists.

        Arguments:
        - 'metric_names' - Names of the initially registered metrics.
        - 'save_metric' - Metric used in generation of save names.
        """
        self.metric_names = metric_names
        self.save_metric = save_metric
        self.step_metrics : Dict[str, List[float]] = {}
        self.epoch_metrics : Dict[str, List[float]] = {}
        self.collections = {
            TrainMetrics.EPOCH_NAME : self.epoch_metrics,
            TrainMetrics.STEP_NAME : self.step_metrics,
        }
        self.registerMetric(self.metric_names)
        
    def _registerSingleMetric(self, metric_name):
        """
        Registers a single loss or metric.

        Arguments:
        - 'metric_name' - The registered name.
        """
        self.collections[TrainMetrics.EPOCH_NAME][metric_name] = []
        self.collections[TrainMetrics.STEP_NAME][metric_name] = []

    def registerMetric(self, metric_name : Union[str, Sequence[str]]) -> None:
        """
        Registers one or a sequence of metrics.

        Arguments:
        - 'metric_name' - A single name or a sequence of names for registration.
        """
        if isinstance(metric_name, str):
            self._registerSingleMetric(metric_name)
        else:
            for name in metric_name:
                self._registerSingleMetric(name)

    def setSaveMetric(self, save_metric : str = None) -> None:
        """
        Sets the name of the metric used for save name generation.

        Arguments:
        - 'save_metric' - The name of an already registered metric.
        """
        self.save_metric = save_metric

    def startEpoch(self) -> None:
        """Initialises step counting for an epoch."""
        self.current_epoch_metrics : Dict[str, List[float]] = {}
        for name in self.metric_names:
            self.current_epoch_metrics[name] = []

    def step(self, current_step_metrics : Dict[str, float]) -> None:
        """
        Updates the steps collections with the provided emtrics.

        Arguments:
        - 'current_step_metrics' - Dictionary of registered names and step values, which should be added to the metric tracker.
        """
        for name, value in current_step_metrics.items():
            self.current_epoch_metrics[name].append(value)
            self.step_metrics[name].append(value)

    def endEpoch(self, val_metrics : Dict[str, float]) -> None:
        """
        Ends the epoch by aggregating values counted during the epoch and sets values to additional supplied metrics.
        'val_metrics' can be validation metrics, which are not counted after each batch (step).

        Arguments:
        - 'val_metrics' - Values for registered metrics, which were not counted in 'step's.
        """
        self.epoch_results : Dict[str, float] = {}
        for name in self.metric_names:
            if len(self.current_epoch_metrics[name]) > 0:
                self.epoch_results[name] = np.mean(self.current_epoch_metrics[name])
                self.epoch_metrics[name].append(self.epoch_results[name])
        for name, value in val_metrics.items():
            self.epoch_metrics[name].append(value)
            self.epoch_results[name] = value
    
    def saveName(self, current_epoch : int, name : str = None) -> str:
        """
        Creates a string checkpoint name from the saved metric that contains the date and a given epoch.

        Arguments:
        - 'current_epoch' - The epoch number, which should be written in the name.
        - 'name' - Short string the name will start with.

        Returns:
        - String with date, epoch and a metric value if set.
        """
        name = "cnn" if name is None else name
        dt = datetime.now()
        eval_name_part = "" if self.save_metric is None else "_l{:.6f}".format(self.epoch_results[self.save_metric])
        return "{}_{:%Y%m%d}_e{:04d}{}".format(name, dt, current_epoch, eval_name_part)

    def saveMetrics(self, target_dir : Path):
        """
        Saves the epoch and step metrics into two files in the given directory.

        Arguments:
        - 'target_dir' - Directory where the metrics files should be saved.
        """
        for key, value in self.collections.items():
            file_name = Path(target_dir, "{}_metrics.txt".format(key))
            saved_names = []
            saved_values = []
            for metric_name, metric_values in value.items():
                if len(metric_values) > 0:
                    saved_names.append(metric_name)
                    saved_values.append(metric_values)
            saved_values = np.reshape(np.asarray(saved_values).T, (-1, len(saved_names)))
            with open(file_name, "w") as metrics_file:
                for i in range(len(saved_names) - 1):
                    metrics_file.write("{} ".format(saved_names[i]))
                metrics_file.write("{}\n".format(saved_names[-1]))
                for idx in range(saved_values.shape[0]):
                    metrics_file.write("{}".format(idx + 1))
                    for value in saved_values[idx]:
                        metrics_file.write(" {}".format(value))
                    metrics_file.write("\n")

    def loadMetrics(self, target_dir : Path):
        """
        Loads the metrics saved in the files in the given directory.

        Arguments:
        - 'target_dir' - Directory where the saved metrics files lie.
        """
        for key, value in self.collections.items():
            file_name = Path(target_dir, "{}_metrics.txt".format(key))
            with open(file_name, "r") as metrics_file:
                lines = metrics_file.readlines()
            metric_names = lines[0].split()
            for name in metric_names:
                if name not in value.keys():
                    value[name] = []
            for temp_row_idx in range(len(lines) - 1):
                row_idx = temp_row_idx + 1
                values = lines[row_idx].split()[1:]
                for column_idx in range(len(values)):
                    value[metric_names[column_idx]].append(float(values[column_idx]))

    def getValues(self, collection_name : str, metric_name : str) -> List[float]:
        """
        Returns the list of values stored in the given collection under the given metric name.

        Arguments:
        - 'collection_name' - Name of the collection: 'epoch' or 'step'.
        - 'metric_name' - A registered name of the requested set of values.

        Returns:
        - List of values stored under the given names.
        """
        return self.collections[collection_name][metric_name]

    def isActive(self, collection_name : str, metric_name : str):
        """
        Returns whether the registered metric is active - values are being added to it.

        Arguments:
        - 'collection_name' - Name of the collection: 'epoch' or 'step'.
        - 'metric_name' - A registered name of queried metric.

        Returns:
        - Boolean value saying whether the metric is active.
        """
        if metric_name in self.collections[collection_name].keys():
            return len(self.collections[collection_name][metric_name]) > 0
        else:
            return False

class MetricCounter:
    """
    Class which helps sum values of a set of metrics and compute their means.
    """

    def __init__(self, metric_names : Sequence[str] = []) -> None:
        """
        Initialises the metric counter and registers initial metrics.

        Arguments:
        - 'metric_names' - A set of initially registered metrics.
        """
        self.metric_counts = {}
        self.added_steps = 0
        self.registerMetric(metric_names)
        self.active_metrics = set()
        
    def _registerSingleMetric(self, metric_name) -> None:
        """
        Registered a single metric counter.

        Arguments:
        - 'metric_name' - Name of the registered metric.
        """
        self.metric_counts[metric_name] = 0

    def registerMetric(self, metric_name : Union[str, Sequence[str]]) -> None:
        """
        Registers one or more metrics for counting.

        Arguments:
        - 'metric_name' - One or more metric names for registration.
        """
        if isinstance(metric_name, str):
            self._registerSingleMetric(metric_name)
        else:
            for name in metric_name:
                self._registerSingleMetric(name)

    def addStep(self, metrics : Dict[str, float]) -> None:
        """
        Adds a single step to the counter - increases the number of counted steps and adds the provided values to the counters.

        Arguments:
        - 'metrics' - Dictionary of registered names and added values.
        """
        for key, value in metrics.items():
            self.metric_counts[key] += value
            self.active_metrics.add(key)
        self.added_steps += 1

    def getValue(self, metric_name : str, mean : bool = False) -> float:
        """
        Returns the counted value, either absolute or mean.

        Arguments:
        - 'metric_name' - Name of the requested metric.
        - 'mean' - Whether the returned value should be the mean of counted values or their sum.

        Returns:
        - The computed value.
        """
        return self.metric_counts[metric_name] / self.added_steps if mean else self.metric_counts[metric_name]
    
    def getActiveValues(self, mean : bool = False) -> Dict[str, float]:
        """
        Returns a dictionary of active values - values which are actively counted.

        Arguments:
        - 'mean' - Whether to return mean values.

        Returns:
        - The computed dictionary.
        """
        active_values = {}
        for metric_name in self.active_metrics:
            active_values[metric_name] = self.metric_counts[metric_name] / self.added_steps if mean else self.metric_counts[metric_name]
        return active_values

    def reset(self) -> None:
        """Resets the counters of this object."""
        for key in self.metric_counts.keys():
            self.metric_counts[key] = 0
        self.added_steps = 0
        self.active_metrics = set()
