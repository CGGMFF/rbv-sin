
from typing import Dict, Tuple, Sequence, Union
from pathlib import Path

import numpy as np
from numpy import ndarray
import skimage.transform
import skimage.io
import tensorflow as tf

from rbv_sin.utils import augment
from rbv_sin.nn.seg_network import VesselSegNetwork
from rbv_sin.nn.custom_objects import MaskIoU
from rbv_sin.nn.network_trainer import NetworkTrainer

class VesselSegmentationTrainer(NetworkTrainer):
    """Implementation of the network trainer for blood vessel segmentation."""

    CP_MODEL_NAME = "seg_weights"

    def __init__(self, network_inputs: Union[Sequence[Tuple[int, ...]], Tuple[int, ...], Dict[str, Tuple[int, ...]]]) -> None:
        """
        Initialises the model with the given input shape.

        Arguments:
        - 'network_inputs' - The input shape for the network.
        """
        self.loss_name = "bin_ce"
        self.iou_name = "iou"
        self.val_loss_name = "val_bin_ce"
        self.val_iou_name = "val_iou"
        self.batch_norm = True
        super().__init__(network_inputs, [self.loss_name, self.iou_name, self.val_loss_name, self.val_iou_name])

    def _defineModel(self) -> None:
        """Implementation of the model definition."""
        self.base_filters = 32
        self.model = VesselSegNetwork(self.network_inputs, self.base_filters, batch_norm=self.batch_norm, trainable=True)

    def saveMetricName(self) -> str:
        """The save metric for vessel segmentation is validation IoU metric."""
        return self.val_iou_name

    def compileModel(self, batch_size: int = 1, epochs: int = 1, batches_per_epoch: int | None = None, cp_epoch_delta: int = 1, start_lr : float = 0.0001, end_lr : float = 0.00001,
                     train_size : int = None, augmentor : augment.Augmentor = None, augmentations : Sequence[int] = None) -> None:
        """
        Implementation of vessel-segmentation-specific model compilation. Has to be called before fitting, not necessary before inference.
        Defines the learning rates, losses and optimisers.

        Arguments:
        - 'batch_size' - The training batch size.
        - 'epochs' - The number of epochs in the upcomming training run.
        - 'batches_per_epoch' - Number of batches sampled in one epoch - None for automatic.
        - 'cp_epoch_delta' - How many epochs have to pass before saving a chcekpoint.
        - 'start_lr' - The start learning rate for exponential decay.
        - 'end_lr' - The end learning rate for exponential decay.
        - 'train_size' - The number of trainign samples.
        - 'augmentor' - The object responsible for sample and batch augmentation.
        - 'augmentations' - The list of requested augmentations.
        """
        super().compileModel(batch_size, epochs, batches_per_epoch, cp_epoch_delta)
        self.compile_train_size = train_size
        self.augmentor = augmentor
        self.augmentations = augmentations

        learning_rate_decay_factor = (end_lr / start_lr)**(1/epochs)
        steps_per_epoch = int(train_size/batch_size)

        self.segmentator_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=start_lr,
            decay_steps=steps_per_epoch,
            decay_rate=learning_rate_decay_factor,
            staircase=True
        )

        self.segmentation_loss = tf.keras.losses.BinaryCrossentropy()

        self.segmentator_optimizer = tf.keras.optimizers.Adam(self.segmentator_lr_schedule)

        self.segmentation_iou = MaskIoU(self.network_inputs[:2])

    def _trainStep(self, batch_images: tf.Tensor, batch_labels: tf.Tensor) -> Dict[str, float]:
        """
        Implementation of a training step on one batch.

        Arguments:
        - 'batch_images' - Model inputs in the required format.
        - 'batch_labels' - Labels in the required format.

        Returns:
        - A dictionary of computed losses and metrics for this batch - binary cross-entropy and IoU.
        """
        batch_size = max(1, batch_images.shape[0])
        with tf.GradientTape() as seg_tape:
            segmented_vessels = self.model(batch_images, training=True)

            seg_loss = self.segmentation_loss(np.expand_dims(batch_labels, -1), segmented_vessels)

            seg_iou = 0
            for label, vessel in zip(batch_labels, segmented_vessels):
                self.segmentation_iou.reset_state()
                self.segmentation_iou.update_state(label, vessel)
                seg_iou += self.segmentation_iou.result().numpy()
            seg_iou /= batch_size

        gradients = seg_tape.gradient(seg_loss, self.model.trainable_variables)
        self.segmentator_optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        seg_iou = self.segmentation_iou.result()

        return {
            self.loss_name : seg_loss.numpy(),
            self.iou_name : seg_iou.numpy(),
        }
    
    def _transformBatch(self, batch_from_streamer: Sequence[ndarray]) -> Tuple[ndarray, ndarray]:
        """
        Applies augmentations to the sampled batch and produces the required inputs and labels..

        Arguments:
        - 'batch_from_streamer' - One batch sampled from the dataset.

        Returns:
        - Model inputs for one training step.
        - Labels for one trainign step.
        """
        augmented_batch = self.augmentor.augmentSamplesMulti(self.augmentations, [[b_img, b_vessels] for (b_img, b_vessels) in zip(batch_from_streamer[0], batch_from_streamer[1])], [False, True])
        batch_images = np.asarray([b[0] for b in augmented_batch])
        batch_labels = np.asarray([b[1] for b in augmented_batch])
        return batch_images, batch_labels
    
    def _validateEpoch(self, validation_images, validation_labels) -> Dict[str, float]:
        """
        Implementation of the model validation. Computes the mean binary cross-entropy and IoU for the validation set.

        Arguments:
        - 'validation_images' - Model validation inputs in the required format.
        - 'validation_labels' - Validation labels i nthe required format.

        Returns:
        - A dictionary with losses and metrics - bin CE and IoU.
        """
        val_result = self.predict(validation_images)
        loss = 0.0
        iou = 0.0
        for gt, res in zip(validation_labels, val_result):
            loss += tf.keras.losses.BinaryCrossentropy()(np.expand_dims(gt, -1), res)
            self.segmentation_iou.reset_state()
            self.segmentation_iou.update_state(gt, res)
            iou += self.segmentation_iou.result().numpy()
        loss /= len(val_result)
        iou /= len(val_result)
        return { self.val_loss_name : loss, self.val_iou_name : iou }
    
    def fit(self, images: ndarray, vessels: ndarray, val_images: ndarray = None, val_labels: ndarray = None) -> None:
        # Print out warning if the training size passed in compilation is different than the size of training data.
        if self.compile_train_size != images.shape[0]:
            print("WARNING: The model is starting trainning with different data size than what was indicated during model compilation.")
        super().fit(images, vessels, val_images, val_labels)
    
    def _predictBatch(self, images: ndarray) -> ndarray:
        """
        Implementation of a single batch prediction - simply calls the segmentation model.

        Arguments:
        - 'images' - Model inputs in the required format.
        """
        return self.model(images, training=False)
    
    def setCheckpointParams(self, cp_path_train_dir: Path = None, cp_train_source : np.ndarray = None, cp_labels : np.ndarray = None, cp_show : bool = True) -> None:
        """
        Sets the checkpoint parameters for segmentation.

        Arguments:
        - 'cp_path_train_dir' - Checkpoint directory path.
        - 'cp_train_source' - Model inputs for the exampels evaluated in the checkpoint.
        - 'cp_labels' - Labels for the examples.
        - 'cp_show' - Whether training graphs should be showed on top of being saved to disk.
        """
        super().setCheckpointParams(cp_path_train_dir, "seg")
        self.cp_train_source = cp_train_source
        self.cp_labels = cp_labels
        self.cp_show = cp_show
        self.cp_examples = (self.cp_train_source is not None) and (self.cp_labels is not None)

    def _saveModels(self, model_dir_path: Path) -> None:
        """Saves the model weights in the given directory."""
        self.model.save_weights(Path(model_dir_path, VesselSegmentationTrainer.CP_MODEL_NAME))

    def _loadModels(self, model_dir_path: Path) -> None:
        """Loads the model weights from the given directory."""
        self.model.load_weights(Path(model_dir_path, VesselSegmentationTrainer.CP_MODEL_NAME))

    def _saveExamples(self, cp_dir: Path) -> None:
        """
        Saves the example results of the model in the given directory. Creates an 'examples' sub-directory.

        Arguments:
        - 'cp_dir' - Checkpoint directory.
        """
        if self.cp_examples:
            sample_result = self.predict(self.cp_train_source)
            examples_dir = Path(cp_dir, "examples")
            Path.mkdir(examples_dir, exist_ok=True)
            for i, (sample, result, label) in enumerate(zip(self.cp_train_source, sample_result, self.cp_labels)):
                sample, result, label = np.squeeze(sample), np.squeeze(result), np.squeeze(label)
                bin_result = result >= 0.5
                bin_label = label >= 0.5
                iou = np.sum(np.logical_and(bin_result, bin_label)) / np.sum(np.logical_or(bin_result, bin_label))
                self.segmentation_iou.reset_state()
                self.segmentation_iou.update_state(label, result)
                skimage.io.imsave(Path(examples_dir, "{:03d}_sample.png".format(i)), np.asarray(sample * 255, np.uint8))
                skimage.io.imsave(Path(examples_dir, "{:03d}_result.png".format(i)), np.asarray(result * 255, np.uint8))
                skimage.io.imsave(Path(examples_dir, "{:03d}_label.png".format(i)), np.asarray(label * 255, np.uint8))
    
    def _saveGraphs(self, cp_dir: Path) -> None:
        """
        Saves training loss and metric graphs in a sub-directory 'graphs'.

        Arguments:
        - 'cp_dir' - Checkpoint directory.
        """
        graphs_dir = Path(cp_dir, "graphs")
        Path.mkdir(graphs_dir, exist_ok=True)
        colors = ["blue", "red"]
        # TODO: Plots
        metric_names = [self.loss_name, self.val_loss_name]
        self._plotMetrics("epoch", metric_names, "Epoch loss", ["Train seg. loss", "Val. seg. loss"], colors, Path(graphs_dir, "epoch_seg_loss.png"), show=self.cp_show)
        metric_names = [self.iou_name, self.val_iou_name]
        self._plotMetrics("epoch", metric_names, "Epoch IoU", ["Train iou", "Val. iou"], colors, Path(graphs_dir, "epoch_iou.png"), np.linspace(0.0, 1.0, 11, endpoint=True), (-0.05, 1.05), show=self.cp_show)
        self._plotMetrics("step", [self.loss_name], "Step loss", ["Train seg. loss"], ["blue"], Path(graphs_dir, "step_seg_loss.png"), show=self.cp_show)
        self._plotMetrics("step", [self.iou_name], "Step iou", ["Train iou"], ["blue"], Path(graphs_dir, "step_iou.png"), np.linspace(0.0, 1.0, 11, endpoint=True), (-0.05, 1.05), show=self.cp_show)

    def numberOfParameters(self) -> Dict[str, int]:
        """Returns a dictionary with the number of parameters under the key 'model'."""
        _ = self.model(tf.zeros([1] + list(self.network_inputs)))
        return { "model" : np.sum([np.prod(v.shape) for v in self.model.trainable_variables]) }
