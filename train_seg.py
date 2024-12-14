
import argparse
from pathlib import Path
import shutil
import numpy as np
import skimage.util
from sklearn.model_selection import train_test_split

from rbv_sin.data import RFIIO
from rbv_sin.utils import augment
from rbv_sin.nn import seg_trainer

parser = argparse.ArgumentParser()
parser.add_argument("--validation", default=0, type=float, help="The percentage of data used for validation. Zero means no validation.")
parser.add_argument("--path_train", default=None, type=str, help="Path to the training directory of the segmentation network. It will be cleaned if the training is not continued.")
parser.add_argument("--path_data", default=None, type=str, help="Path to the data file for training.")
parser.add_argument("--continuation", default=None, type=str, help="Checkpoint name which should be continued or None if the training is fresh.")
parser.add_argument("--show_cp_graphs", default=False, action="store_true", help="Whether the training loss graphs should be showed during checkpoint save.")
parser.add_argument("--batch_size", default=16, type=int, help="The training batch size.")
parser.add_argument("--epochs", default=100, type=int, help="The number of training epochs.")
parser.add_argument("--cp_epoch_delta", default=5, type=int, help="The number of epoch that has to pass between checkpoints.")
parser.add_argument("--start_lr", default=0.001, type=float, help="The start learning rate.")
parser.add_argument("--end_lr", default=0.0001, type=float, help="The end learning rate.")

parser.add_argument("--val_seed", default=42, type=int, help="The rng seed for validation splitting.")
parser.add_argument("--aug_seed", default=1169, type=int, help="The rng seed for data augmentation.")

def main(args : argparse.Namespace):
    print("Loading the training data: '{}'.".format(args.path_data), end="\r")
    rfi_io = RFIIO()
    rfi_train_set = rfi_io.load(Path(args.path_data))

    image_data = np.asarray([skimage.util.img_as_float(sample.image) for sample in rfi_train_set])
    vessel_data = np.asarray([skimage.util.img_as_float(sample.vessel_mask) for sample in rfi_train_set])
    print("Finished loading training data: '{}'.".format(args.path_data))

    # Split the data for validation if it is indicated in the arguments.
    if args.validation > 0:
        vessels_train, vessels_validation, images_train, images_validation = train_test_split(vessel_data, image_data, test_size=args.validation, random_state=args.val_seed)
    else:
        vessels_train, images_train, vessels_validation, images_validation = vessel_data, image_data, None, None

    # Hard-coded network input shape. The network is not limited to this input shape but it was used in the article.
    network_inputs = [128, 128, 3]

    augmentor = augment.Augmentor(rng=args.aug_seed)
    augmentations = [augment.Augmentor.MIRROR_V, augment.Augmentor.MIRROR_H, augment.Augmentor.ROTATION, augment.Augmentor.ZOOM, augment.Augmentor.BRIGHTNESS, augment.Augmentor.CONTRAST]
    network = seg_trainer.VesselSegmentationTrainer(network_inputs)
    network.compileModel(args.batch_size, args.epochs, None, args.cp_epoch_delta, args.start_lr, args.end_lr, vessels_train.shape[0], augmentor, augmentations)
    print("Segmentation network number of parameters: {}.".format(network.numberOfParameters()))

    path_train = Path(args.path_train)
    if args.continuation is not None:
        cp_path = Path(path_train, args.continuation)
        network.loadCheckpoint(cp_path)
    else:
        if Path.exists(path_train):
            shutil.rmtree(path_train)
        Path.mkdir(path_train, parents=True)

    network.setCheckpointParams(path_train, images_validation, vessels_validation, cp_show=args.show_cp_graphs)
    print("Starting segmentation training...")
    network.fit(images_train, vessels_train, images_validation, vessels_validation)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
