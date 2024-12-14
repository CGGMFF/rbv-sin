
import time
from typing import Dict

class TrainReport:
    """Class for reporting the training progress."""

    def __init__(self) -> None:
        """Initialises the train reporter. The training algorithm should print anything if TrainReporter is active."""
        self.epoch_running = False
        self.bar_symbols = 10
        self.bar_passed = "-"
        self.bar_todo = " "

    def startEpoch(self, epochs_current : int, epochs_total : int, epochs_offset : int, batches_per_epoch : int) -> None:
        """
        Initialises and starts the reporting of one epoch.

        Arguments:
        - 'epochs_current' - Current number of epoch in the reported run.
        - 'epochs_total' - The total number of epochs in the reported run.
        - 'epochs_offset' - The number of epochs completed before beginning this report - previously completed training run.
        - 'batches_per_epoch' - Number of batches in a single epoch.
        """
        if self.epoch_running:
            self.cancelEpoch()
            print("WARNING: Starting an epoch without calling 'endEpoch' for the previous one - the previous epoch was cancelled!")
        self.epoch_running = True
        self.start_time = time.time()
        self.epochs_total = epochs_total
        self.epochs_offset = epochs_offset
        self.batches_per_epoch = batches_per_epoch
        self.epochs_current = epochs_current
        self.batch_current = 0
        self.train_metrics = {}
        self.epoch_symbol_count = len(str(epochs_total))
        self.batch_symbol_count = len(str(batches_per_epoch))
        self._report()

    def updateEpoch(self, batch_current : int, train_metrics : Dict[str, float]) -> None:
        """
        Updates the reporter progress and prints out the given metric values.

        Arguments:
        - 'batch_current' - Current number of batch in an epoch.
        - 'train_metrics' - A dictionary of reported metrics.
        """
        self.batch_current = batch_current
        self.train_metrics = train_metrics
        self._report()

    def pauseEpoch(self) -> None:
        """Pauses the reporting by printing a newline - this should be called if one needs to print something else. It is resumed by calling another update."""
        self._report(report_end="\n")

    def endEpoch(self, val_metrics : Dict[str, float]) -> None:
        """
        Ends epoch by reporting additional values not reported in updates.
        These are inteded to be validation metrics - they will be separated by '; VAL:' from the update metrics.

        Arguments:
        - 'val_metrics' - Metrics reported at the end of the epoch.
        """
        self._report(report_end="")
        if len(val_metrics.keys()) > 0:
            print("; VAL: {}".format(self._metricReport(val_metrics)), end="\n")
        else:
            print("", end="\n")
        self.epoch_running = False

    def cancelEpoch(self) -> None:
        """Cancels the epoch reporting. It cannot be resumed and a new epoch should be started."""
        self._report(report_end="\n")
        self.epoch_running = False

    def _metricReport(self, metrics : Dict[str, float]) -> str:
        """
        Creates a string with reported metrics.

        Arguments:
        - 'metrics' - A dictionary of the reported metrics.

        Returns:
        - String with concatenated metric values separated by commas.
        """
        report = ""
        for key, value in metrics.items():
            if len(report) > 0:
                report += ", "
            report += "{}: {:>8.5f}".format(key, value)
        return report

    def _report(self, report_end : str = "\r") -> None:
        """
        Reports the epoch status by printing a formatted string.

        Arguments:
        - 'report_end' - The escape character which ends the report: '\\r' or '\\n'.
        """
        metric_report = self._metricReport(self.train_metrics)
        batch_elapsed = self.batch_current / self.batches_per_epoch
        batch_completed = round(batch_elapsed * self.bar_symbols, None)
        batch_todo = self.bar_symbols - batch_completed
        current_time = time.time()
        print("Epoch {:>{epoch_count}}/{:<{epoch_count}}, {:8.4f} sec, [{}{}] {:>{batch_count}}/{:<{batch_count}}; {}".format(
            self.epochs_offset + self.epochs_current, self.epochs_offset + self.epochs_total, current_time - self.start_time, batch_completed * self.bar_passed,
            batch_todo * self.bar_todo, self.batch_current, self.batches_per_epoch, metric_report, epoch_count=self.epoch_symbol_count, batch_count=self.batch_symbol_count
        ), end=report_end)
