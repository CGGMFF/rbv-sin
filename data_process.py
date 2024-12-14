
import argparse
from pathlib import Path
import numpy as np

from rbv_sin.data import RFIIO, RFISet, RFI, ChaseDB1Loader, FivesLoader
from rbv_sin.utils import transform, dataset, image_splitter

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="fives", type=str, help="The dataset selected for processing.")
parser.add_argument("--path_data", default="", type=str, help="Path to the root directory of the dataset indicated in '--dataset'.")
parser.add_argument("--no_filter", default=False, action="store_true", help="Do not filter the FIVES dataset by removing the erroneous image.")
parser.add_argument("--path_roi", default=None, type=str, help="Path to the directory with dataset region of interests.")
parser.add_argument("--path_save", default=None, type=str, help="Path to the directory where the results of data processing should be saved.")
parser.add_argument("--patch_shape", default=128, type=int, help="The square side of requested patches.")
parser.add_argument("--patch_stride", default=32, type=int, help="Stride at which the patches are extracted. We use 32 for 'chase' and 64 for 'fives'.")

def generatePatches(rfi_io : RFIIO, args : argparse.Namespace):
    splitter = image_splitter.ImageSplitter((args.patch_shape, args.patch_shape), args.patch_stride)
    rfi_set = rfi_io.load(Path(args.path_data))

    rfi_patch_set = []
    patch_idx = 0
    for sample in rfi_set:
        image, vessels = sample.image, sample.vessel_mask
        image_patches = splitter.split(image)
        vessel_patches = splitter.split(vessels)
        for image, vessels in zip(image_patches, vessel_patches):
            rfi = RFI("patch_{:04d}".format(patch_idx))
            rfi.image = image
            rfi.vessel_mask = vessels
            rfi_patch_set.append(rfi)
            patch_idx += 1
       
    rfi_patch_set = RFISet("{}_patches_{}_{}".format(rfi_set.name, args.patch_shape, args.patch_stride), rfi_patch_set)
    rfi_io.store(Path(args.path_save, "{}_patches_{}_{}.npz".format(Path(args.path_data).stem, args.patch_shape, args.patch_stride)), rfi_patch_set)

def processFives(rfi_io : RFIIO, resizer : transform.RFIResize, args : argparse.Namespace):
    loader = FivesLoader(Path(args.path_data))
    train_set, test_set = loader.getSet("train"), loader.getSet("test")
    train_set, train_rois = dataset.applyRoi(train_set, Path(args.path_roi, "fives_train"))
    test_set, test_rois = dataset.applyRoi(test_set, Path(args.path_roi, "fives_test"))
    train_set = resizer.apply(train_set)
    test_set = resizer.apply(test_set)
    if not args.no_filter:
        # Removing the erroneous image (where the mask doesn't fit the image) from the dataset.
        train_set = dataset.SampleFilter().removeNames(train_set, ["174_D"])
    rfi_io.store(Path(args.path_save, "fives_train_256.npz"), train_set)
    rfi_io.store(Path(args.path_save, "fives_test_256.npz"), test_set)

def processChase(rfi_io : RFIIO, resizer : transform.RFIResize, args : argparse.Namespace):
    loader = ChaseDB1Loader(Path(args.path_data))
    chase_set = loader.getSet("all")
    chase_set, chase_rois = dataset.applyRoi(chase_set, Path(args.path_roi, "chase"))
    chase_set = resizer.apply(chase_set)
    # Remove the second vessel annotations.
    for sample in chase_set:
        sample.vessel_mask = sample.vessel_mask[0]
    train_set, test_set = chase_set.subset(np.arange(20)), chase_set.subset(np.arange(20, len(chase_set)))
    rfi_io.store(Path(args.path_save, "chase_train_256.npz"), train_set)
    rfi_io.store(Path(args.path_save, "chase_test_256.npz"), test_set)

def main(args : argparse.Namespace):
    resizer = transform.RFIResize((256, 256))
    rfi_io = RFIIO()
    Path.mkdir(Path(args.path_save), exist_ok=True, parents=True)
    if args.dataset.lower() == "fives":
        processFives(rfi_io, resizer, args)
    elif args.dataset.lower() in ["chase", "chasedb1", "chase_db1"]:
        processChase(rfi_io, resizer, args)
    elif args.dataset.lower() in ["patch", "patches"]:
        generatePatches(rfi_io, args)
    else:
        raise ValueError("Unknown 'dataset' mode: '{}'.".format(args.dataset))

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
