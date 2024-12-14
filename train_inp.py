
import argparse
from pathlib import Path
import shutil
import numpy as np
import skimage.util
from sklearn.model_selection import train_test_split

from rbv_sin.data import RFIIO
from rbv_sin.utils import inp_data_gen, mask_gen, augment
from rbv_sin.nn import inp_trainer

parser = argparse.ArgumentParser()
parser.add_argument("--validation", default=0, type=float, help="The percentage of data used for validation. Zero means no validation.")
parser.add_argument("--path_train", default=None, type=str, help="Path to the training directory of the inpainting network. It will be cleaned if the training is not continued.")
parser.add_argument("--path_data", default=None, type=str, help="Path to the data file for training.")
parser.add_argument("--continuation", default=None, type=str, help="Checkpoint name which should be continued or None if the training is fresh.")
parser.add_argument("--show_cp_graphs", default=False, action="store_true", help="Whether the training loss graphs should be showed during checkpoint save.")
parser.add_argument("--batch_size", default=1, type=int, help="The training batch size.")
parser.add_argument("--epochs", default=100, type=int, help="The number of training epochs.")
parser.add_argument("--cp_epoch_delta", default=5, type=int, help="The number of epoch that has to pass between checkpoints.")
parser.add_argument("--start_lr", default=0.001, type=float, help="The start learning rate.")
parser.add_argument("--end_lr", default=0.0001, type=float, help="The end learning rate.")

parser.add_argument("--mask_seed", default=42583, type=int, help="The rng seed for training mask generation.")
parser.add_argument("--val_seed", default=42, type=int, help="The rng seed for validation splitting.")
parser.add_argument("--aug_seed", default=1169, type=int, help="The rng seed for data augmentation.")

parser.add_argument("--oce_sigma", default=13, type=float, help="The gaussian sigma used during optic centre estimation.")
parser.add_argument("--oce_channel", default=0, type=int, help="The image channel used during optic centre estimation.")
parser.add_argument("--focus_radius", default=0.25, type=float, help="The image height fraction used as the focus mask radius.")
parser.add_argument("--focus_density", default=0.5, type=float, help="The required converage of the focus mask.")
parser.add_argument("--val_mask_ext", default=3, type=int, help="Radius of disc structuring element used in morphology mask extension.")

def main(args : argparse.Namespace):
    print("Loading the training data: '{}'.".format(args.path_data), end="\r")
    rfi_io = RFIIO()
    rfi_train_set = rfi_io.load(Path(args.path_data))

    image_data = np.asarray([skimage.util.img_as_float(sample.image) for sample in rfi_train_set])
    vessel_data = np.asarray([skimage.util.img_as_float(sample.vessel_mask) for sample in rfi_train_set])
    print("Finished loading training data: '{}'.".format(args.path_data))

    centre_estimator = inp_data_gen.OpticCentreEstimator(args.oce_sigma, args.oce_channel)
    mask_generator = mask_gen.CustomMaskGenerator(vessel_data[0].shape, args.focus_radius, None, args.focus_density, rng=args.mask_seed)
    data_transformer = inp_data_gen.InpaintDataTransformer(mask_generator, centre_estimator)

    # Split the data for validation if it is indicated in the arguments.
    if args.validation > 0:
        print("Preparing validation data...", end="\r")
        mask_extender_val = mask_gen.MaskMorphology(args.val_mask_ext, "dilation", 0.5)
        images_train, images_validation, vessels_train, vessels_validation = train_test_split(image_data, vessel_data, test_size=args.validation, random_state=args.val_seed)
        source_validation = np.asarray([data_transformer.transform(image, mask) for image, mask in zip(images_validation, vessels_validation)])
        source_blind = np.asarray([data_transformer.transform(image, mask_extender_val(mask), no_generation=True) for image, mask in zip(images_validation, vessels_validation)])
        print("Finished preparing validation data.")
    else:
        images_train = image_data
        vessels_train = vessel_data
        images_validation = None
        source_validation = None
        source_blind = None

    # Hard-coded network input shapes. The network is not limited to these input shapes but they were used in the article.
    network_inputs = { "generator" : [256, 256, 4], "vgg19" : [256, 256, 3] }

    augmentor = augment.Augmentor(rng=args.aug_seed)
    augmentations = [augment.Augmentor.MIRROR_V, augment.Augmentor.MIRROR_H, augment.Augmentor.ROTATION, augment.Augmentor.ZOOM, augment.Augmentor.BRIGHTNESS, augment.Augmentor.CONTRAST]
    network = inp_trainer.VesselInpaintingTrainer(network_inputs)
    network.compileModel(args.batch_size, args.epochs, None, args.cp_epoch_delta, args.start_lr, args.end_lr, images_train.shape[0], augmentor, augmentations, data_transformer)
    print("Inpainting network number of parameters: {}.".format(network.numberOfParameters()))

    path_train = Path(args.path_train)
    if args.continuation is not None:
        cp_path = Path(path_train, args.continuation)
        network.loadCheckpoint(cp_path)
    else:
        if Path.exists(path_train):
            shutil.rmtree(path_train)
        Path.mkdir(path_train, parents=True)

    network.setCheckpointParams(path_train, source_validation, images_validation, source_blind, cp_show=args.show_cp_graphs)
    print("Starting inpainting training...")
    network.fit(images_train, vessels_train, source_validation, images_validation)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
