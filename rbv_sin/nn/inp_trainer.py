
from typing import Dict, Tuple, Sequence, Union
from pathlib import Path

import numpy as np
from numpy import ndarray
import skimage.io
import tensorflow as tf

from rbv_sin.utils import augment
from rbv_sin.utils.inp_data_gen import InpaintDataTransformer
from rbv_sin.nn.inp_network import VesselInpGenerator
from rbv_sin.nn.sin_factory import SINVGG19
from rbv_sin.nn.network_trainer import NetworkTrainer

class VesselInpaintingTrainer(NetworkTrainer):
    """Implementation of the custom network trainer for blood vessel inpainting."""

    CP_GENERATOR_NAME = "inp_generator_weights.h5"

    def __init__(self, network_inputs: Union[Sequence[Tuple[int, ...]], Tuple[int, ...], Dict[str, Tuple[int, ...]]]) -> None:
        """
        Initialises and defines the blood vessel inpainting trainer.

        Arguments:
        - 'network_inputs' - Inputs of the inpainting network. It should define all possible inputs, e.g. the generator and the vgg for perceptual loss.
        """
        self.loss_name = "loss"
        self.rec_loss_name = "rec_loss"
        self.perc_loss_name = "perc_loss"
        self.val_rec_loss_name = "val_rec_loss"
        self.generator_batch_norm = False
        super().__init__(network_inputs, [self.loss_name, self.rec_loss_name, self.perc_loss_name, self.val_rec_loss_name])

    def _defineModel(self) -> None:
        """Implementation of the model definition. Defines both the inapinting generator and the vgg19 network for perceptual loss."""
        self.generator_base_filters = 32
        self.generator = VesselInpGenerator(self.network_inputs["generator"], self.generator_base_filters, batch_norm=self.generator_batch_norm, trainable=True)
        self.vgg19 = SINVGG19(self.network_inputs["vgg19"], trainable=False)

    def saveMetricName(self) -> str:
        """The inpainting saving loss is the validation L1 reconstruction loss."""
        return self.val_rec_loss_name

    def compileModel(self, batch_size: int = 1, epochs: int = 1, batches_per_epoch: int | None = None, cp_epoch_delta: int = 1, start_lr : float = 0.0001, end_lr : float = 0.00001,
                     train_size : int = None, augmentor : augment.Augmentor = None, augmentations : Sequence[int] = None, data_transformer : InpaintDataTransformer = None) -> None:
        """
        Implementation of the inpainting model compilation. This method has to be called before training but not before inference.
        This method defines learning rate decay, loss functions, weights and the optimiser.

        Arguments:
        - 'batch_size' - The training batch size.
        - 'epochs' - The numebr of trainign epochs in the upcoming training run.
        - 'batches_per_epoch' - The number of batches in one epoch for batch sampler - None to determine the number automatically.
        - 'cp_epoch_delta' - The number of epochs between model checkpoint saves.
        - 'start_lr' - The start learning rate for exponential decay.
        - 'end_lr' - The end learning rate for exponential decay.
        - 'train_size' - The size of the trainign data - needed for exponential decay definition.
        - 'augmentor' - Object realising sample and batch augmentation.
        - 'augmentations' - The requested sample augmentations.
        - 'data_transformer' - Object generating inpainting input data from images and masks.
        """
        super().compileModel(batch_size, epochs, batches_per_epoch, cp_epoch_delta)
        self.compile_train_size = train_size
        self.augmentor = augmentor
        self.augmentations = augmentations
        self.data_transformer = data_transformer

        learning_rate_decay_factor = (end_lr / start_lr)**(1/epochs)
        steps_per_epoch = int(train_size/batch_size)

        self.generator_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=start_lr,
            decay_steps=steps_per_epoch,
            decay_rate=learning_rate_decay_factor,
            staircase=True
        )

        # Hard-coded loss weights.
        self.rec_weight = 0.9
        self.perc_weight = 0.1
        self.vgg_layer_weights = tf.constant([0.1, 0.2, 0.3, 0.4])

        self.generator_loss = tf.keras.losses.MeanAbsoluteError()
        self.perceptual_loss = tf.keras.losses.MeanAbsoluteError()
        self.generator_optimizer = tf.keras.optimizers.Adam(self.generator_lr_schedule)

    @tf.function
    def _reconstructionLoss(self, inpainted_images : tf.Tensor, true_images : tf.Tensor) -> float:
        """
        Computes the L1 reconstruction loss.

        Arguments:
        - 'inpainted_images' - Inpainted results of the model.
        - 'true_images' - The ground truth images.

        Returns:
        - The L1 loss value.
        """
        gen_rec_loss = self.generator_loss(true_images, inpainted_images)
        return gen_rec_loss

    @tf.function
    def _perceptualLoss(self, inpainted_images : tf.Tensor, true_images : tf.Tensor) -> float:
        """
        Computes the perceptual loss.

        Arguments:
        - 'inpainted_images' - Inpainted results of the model.
        - 'true_images' - The ground truth images.

        Returns:
        - The perceptual loss value.
        """
        inp_out = self.vgg19(inpainted_images)
        true_out = self.vgg19(true_images)
        perc_loss = tf.math.reduce_sum([self.vgg_layer_weights[idx] * self.perceptual_loss(true_tensor, inp_tensor) for idx, (true_tensor, inp_tensor) in enumerate(zip(true_out, inp_out))])
        return perc_loss
    
    def _trainStep(self, batch_images: tf.Tensor, batch_labels: tf.Tensor) -> Dict[str, float]:
        """
        Implementation of a single inpainting training step. Computes the update on one batch.

        Arguments:
        - 'batch_images' - The model inputs for one batch.
        - 'batch_labels' - The expected outputs (labels) for one batch.

        Returns:
        - A dictionary with the loss values - L1, perceptual and combined.
        """
        with tf.GradientTape() as gen_tape:
            inpainted_images = self.generator(batch_images, training=True)

            gen_rec_loss = self._reconstructionLoss(inpainted_images, batch_labels)
            perc_loss = self._perceptualLoss(inpainted_images, batch_labels)
            gen_loss = gen_rec_loss * self.rec_weight + perc_loss * self.perc_weight

        gradients_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_generator, self.generator.trainable_variables))

        return {
            self.loss_name : gen_loss.numpy(),
            self.rec_loss_name : gen_rec_loss.numpy(),
            self.perc_loss_name : perc_loss.numpy(),
        }
    
    def _transformBatch(self, batch_from_streamer: Sequence[ndarray]) -> Tuple[ndarray, ndarray]:
        """
        Transforms one batch sampled from the dataset into the requried model inputs.
        It applies augmentations to the batch samples and custom inpainting input transformation.

        Arguments:
        - 'batch_from_streamer' - A batch sampled from the dataset.

        Returns:
        - The expected model inputs computed from the batch.
        - The expected model outputs (labels) for the batch.
        """
        augmented_batch = self.augmentor.augmentSamplesMulti(self.augmentations, [[b_img, b_vessels] for (b_img, b_vessels) in zip(batch_from_streamer[0], batch_from_streamer[1])], [False, True])
        batch_images = self.data_transformer.transformPairedBatch(augmented_batch)
        batch_labels = np.asarray([b[0] for b in augmented_batch])
        return batch_images, batch_labels
    
    def _validateEpoch(self, validation_images, validation_labels) -> Dict[str, float]:
        """
        Implementation of the inpainting model validation. Predicts the validation input data and computes the mean L1 reconstruction loss.

        Arguments:
        - 'validation_images' - The expected validation model inputs.
        - 'validation_labels' - The expected validation model outputs (labels).

        Returns:
        - A dictionary of the computed metrics - mean L1 loss in this case.
        """
        validation_result = self.predict(validation_images, batch_size=2)
        loss = 0.0
        for gt, res in zip(validation_labels, validation_result):
            loss += self.generator_loss(gt, res)
        loss /= len(validation_result)
        return {self.val_rec_loss_name : loss}
    
    def fit(self, images: ndarray, vessels: ndarray, val_images: ndarray = None, val_labels: ndarray = None) -> None:
        # Print warning if the trainign dataset has different size than what was indicated in the compilation.
        if self.compile_train_size != images.shape[0]:
            print("WARNING: The model is starting trainning with different data size than what was indicated during model compilation.")
        super().fit(images, vessels, val_images, val_labels)
    
    def _predictBatch(self, images: ndarray) -> ndarray:
        return self.generator(images, training=False)
    
    def setCheckpointParams(self, cp_path_train_dir: Path = None, cp_train_source : np.ndarray = None, cp_labels : np.ndarray = None, cp_target_source : np.ndarray = None, cp_show : bool = True) -> None:
        """
        Sets the checkpoint parameters for the inpainting model.

        Arguments:
        - 'cp_path_train_dir' - Path to the checkpoint directory.
        - 'cp_train_source' - The expected model inputs for a checkpoint example dataset.
        - 'cp_labels' - The labels for 'cp_train_source'.
        - 'cp_target_source' - The expected inputs for examples without the ground truth - what we actually intend to inpaint.
        - 'cp_show' - Whether to show the training loss graphs during checkpoint saving.
        """
        super().setCheckpointParams(cp_path_train_dir, "inp")
        self.cp_train_source = cp_train_source
        self.cp_labels = cp_labels
        self.cp_target_source = cp_target_source
        self.cp_show = cp_show
        self.cp_examples = (self.cp_train_source is not None) and (self.cp_target_source is not None) and (self.cp_labels is not None)

    def _ensureGeneratorName(self, model_name : str) -> str:
        """Returns the default name if 'model_name' is None."""
        return VesselInpaintingTrainer.CP_GENERATOR_NAME if model_name is None else model_name

    def _saveModels(self, model_dir_path: Path, model_name : str = None) -> None:
        """Saves the model weights in the given directory."""
        self.generator.save_weights(Path(model_dir_path, self._ensureGeneratorName(model_name)))

    def _loadModels(self, model_dir_path: Path, model_name : str = None) -> None:
        """Loads the model weights from the given directory."""
        # Run the model to build connections before loading the weights (this is necessary in some/newer versions).
        _ = self.generator(tf.zeros([1] + list(self.network_inputs["generator"])))
        self.generator.load_weights(Path(model_dir_path, self._ensureGeneratorName(model_name)))

    def _saveExamples(self, cp_dir: Path) -> None:
        """
        Saves the example results of the model in the given directory. Creates an 'examples' sub-directory.

        Arguments:
        - 'cp_dir' - Checkpoint directory.
        """
        if self.cp_examples:
            sample_result = self.predict(self.cp_train_source)
            target_result = self.predict(self.cp_target_source)
            examples_dir = Path(cp_dir, "examples")
            Path.mkdir(examples_dir, exist_ok=True)
            for i, (sample, result, label) in enumerate(zip(self.cp_target_source, sample_result, self.cp_labels)):
                result, label, sample = np.clip(result, 0.0, 1.0), np.clip(label, 0.0, 1.0), np.clip(sample, 0.0, 1.0)
                skimage.io.imsave(Path(examples_dir, "{:03d}_result.png".format(i)), np.asarray(result * 255, np.uint8))
                skimage.io.imsave(Path(examples_dir, "{:03d}_label.png".format(i)), np.asarray(label * 255, np.uint8))
                skimage.io.imsave(Path(examples_dir, "{:03d}_vessels.png".format(i)), np.asarray(sample[:, :, 3] * 255, np.uint8))
            for i, sample in enumerate(target_result):
                sample = np.clip(sample, 0.0, 1.0)
                skimage.io.imsave(Path(examples_dir, "{:03d}_target.png".format(i)), np.asarray(sample * 255, np.uint8))
    
    def _saveGraphs(self, cp_dir: Path) -> None:
        """
        Saves training loss and metric graphs in a sub-directory 'graphs'.

        Arguments:
        - 'cp_dir' - Checkpoint directory.
        """
        graphs_dir = Path(cp_dir, "graphs")
        Path.mkdir(graphs_dir, exist_ok=True)
        colors = ["blue", "red"]
        metric_names = [self.rec_loss_name, self.val_rec_loss_name]
        metric_labels = ["Train rec. loss", "Val. rec. loss"]
        self._plotMetrics("epoch", metric_names, "Epoch reconstruction loss", metric_labels, colors, Path(graphs_dir, "epoch_rec_loss.png"), show=self.cp_show)
        self._plotMetrics("step", [self.rec_loss_name], "Step reconstruction loss", ["Train rec. loss"], ["blue"], Path(graphs_dir, "step_rec_loss.png"), show=self.cp_show)
        self._plotMetrics("epoch", [self.loss_name], "Epoch train loss", ["Train loss"], ["blue"], Path(graphs_dir, "epoch_loss.png"), show=self.cp_show)
        self._plotMetrics("step", [self.loss_name], "Step train loss", ["Train loss"], ["blue"], Path(graphs_dir, "step_loss.png"), show=self.cp_show)
        self._plotMetrics("epoch", [self.perc_loss_name], "Epoch train perceptual loss", ["Train perc. loss"], ["blue"], Path(graphs_dir, "epoch_perc_loss.png"), show=self.cp_show)
        self._plotMetrics("step", [self.perc_loss_name], "Step train perceptual loss", ["Train perc. loss"], ["blue"], Path(graphs_dir, "step_perc_loss.png"), show=self.cp_show)

    def numberOfParameters(self) -> Dict[str, int]:
        """Returns a dictionary with the number of parameters under the key 'generator'."""
        _ = self.generator(tf.zeros([1] + list(self.network_inputs["generator"])))
        return { "generator" : np.sum([np.prod(v.shape) for v in self.generator.trainable_variables]) }
