
import argparse
from pathlib import Path
import shutil
import numpy as np
import skimage.io
import skimage.util
import skimage.transform
import matplotlib.pyplot as plt

from rbv_sin.data import RFISet, RFIIO
from rbv_sin.utils import image_splitter, wrappers, evaluation, aabb_selector

parser = argparse.ArgumentParser()
parser.add_argument("--path_data", default=None, type=str, help="Path to the dataset file for evaluation, this should a preprocessed RFI dataset.")
parser.add_argument("--data_idx", default=None, type=int, help="Index of an image in the dataset for single sample evaluation.")
parser.add_argument("--data_name", default=None, type=str, help="Name of the sample in the dataset for single sample evaluation.")
parser.add_argument("--path_image", default=None, type=str, help="Path to a retinal image for single sample evaluation.")
parser.add_argument("--path_mask", default=None, type=str, help="Path to a retinal blood vessel mask for single sample evaluation.")
parser.add_argument("--suppress_transform", default=False, action="store_true", help="Prevents image cropping and resizing if the example is loaded from an image file.")

parser.add_argument("--cp_seg", default="", type=str, help="Path to the checkpoint of the segmentation network.")
parser.add_argument("--split_stride", default=32, type=int, help="Stride used during image splitting and composition for vessel segmentation.")
parser.add_argument("--threshold", default=0.5, type=float, help="The vessel mask binarisation threshold.")
parser.add_argument("--path_save", default=None, type=str, help="Path to the directory where the result should be saved.")
parser.add_argument("--show_results", default=False, action="store_true", help="Whether to visually show the results of the algorithm.")

def evaluateExample(onh_image : np.ndarray, onh_vessel_mask : np.ndarray, name : str, args : argparse.Namespace) -> bool:
    splitter = image_splitter.ImageSplitter((128, 128), (args.split_stride, args.split_stride))
    seg_wrapper = wrappers.SINSegmentationTileWrapper(args.cp_seg, [128, 128, 3], splitter)

    predicted_mask_prob = seg_wrapper(onh_image)
    predicted_mask_bin = predicted_mask_prob > args.threshold

    if onh_vessel_mask is not None:
        mask_evaluator = evaluation.SegEval(predicted_mask_bin, onh_vessel_mask, predicted_mask_prob, args.threshold)
        print("Blood vessel segmentation performance.")
        print(" - IOU:  {:.4f}".format(mask_evaluator.iou()))
        print(" - F1:   {:.4f}".format(mask_evaluator.f1()))
        print(" - REC:  {:.4f}".format(mask_evaluator.recall()))
        print(" - PREC: {:.4f}".format(mask_evaluator.precision()))
        print(" - SPEC: {:.4f}".format(mask_evaluator.specificity()))
        print(" - ACC:  {:.4f}".format(mask_evaluator.accuracy()))
        print(" - AUC:  {:.4f}".format(mask_evaluator.rocAuc()))

    if args.path_save is not None:
        Path.mkdir(Path(args.path_save), exist_ok=True, parents=True)
        skimage.io.imsave(Path(args.path_save, "{}_source.png".format(name)), np.asarray(np.clip(onh_image, 0, 1) * 255, np.uint8))
        skimage.io.imsave(Path(args.path_save, "{}_vessels_prob.png".format(name)), np.asarray(np.clip(predicted_mask_prob, 0, 1) * 255, np.uint8))
        skimage.io.imsave(Path(args.path_save, "{}_vessels_bin.png".format(name)), np.asarray(np.clip(predicted_mask_bin, 0, 1) * 255, np.uint8))
        if onh_vessel_mask is not None:
            skimage.io.imsave(Path(args.path_save, "{}_vessels_true.png".format(name)), np.asarray(np.clip(onh_vessel_mask, 0, 1) * 255, np.uint8))

    if args.show_results:
        images = [onh_image, onh_vessel_mask, predicted_mask_bin, predicted_mask_prob]
        titles = ["Source ONH image", "True vessel mask", "Predicted vessels (binarised)", "Predicted vessels (probabilities)"]
        valid = np.sum([image is not None for image in images])
        fig, ax = plt.subplots(1, valid, figsize=(valid * 4, 4.5))
        idx = 0
        for image, title in zip(images, titles):
            if image is not None:
                ax[idx].imshow(np.clip(image, 0, 1), cmap="gray")
                ax[idx].set_title(title, fontsize=10)
                ax[idx].axis("off")
                idx += 1
        fig.tight_layout()
        plt.show()

    return True

def evaluateDataset(rfi_set : RFISet, args : argparse.Namespace) -> bool:
    splitter = image_splitter.ImageSplitter((128, 128), (args.split_stride, args.split_stride))
    seg_wrapper = wrappers.SINSegmentationTileWrapper(args.cp_seg, [128, 128, 3], splitter)
    if args.path_save is not None:
        path_save = Path(args.path_save)
        if Path.exists(path_save):
            shutil.rmtree(path_save)
        Path.mkdir(path_save, parents=True)

    set_evaluator = evaluation.SetSegEval(args.threshold)
    for idx, sample in enumerate(rfi_set):
        predicted_mask_prob = seg_wrapper(skimage.util.img_as_float(sample.image))
        predicted_mask_bin = predicted_mask_prob > args.threshold
        set_evaluator.addSample(predicted_mask_bin, skimage.util.img_as_float(sample.vessel_mask), predicted_mask_prob)
        if args.path_save is not None:
            name = sample.name if sample.name is not None else idx
            skimage.io.imsave(Path(args.path_save, "{}_vessels_prob.png".format(name)), np.asarray(np.clip(predicted_mask_prob, 0, 1) * 255, np.uint8))
            skimage.io.imsave(Path(args.path_save, "{}_vessels_bin.png".format(name)), np.asarray(np.clip(predicted_mask_bin, 0, 1) * 255, np.uint8))
            skimage.io.imsave(Path(args.path_save, "{}_vessels_true.png".format(name)), np.asarray(np.clip(skimage.util.img_as_float(sample.vessel_mask), 0, 1) * 255, np.uint8))
        print("Processed image {:3d}/{:<3d}".format(idx + 1, len(rfi_set)), end="\r")
    print()

    print("Mean '{}' dataset results.".format(rfi_set.name))
    print(" - IOU:  {:.4f}".format(set_evaluator.iouMean()))
    print(" - F1:   {:.4f}".format(set_evaluator.f1Mean()))
    print(" - REC:  {:.4f}".format(set_evaluator.recallMean()))
    print(" - PREC: {:.4f}".format(set_evaluator.precisionMean()))
    print(" - SPEC: {:.4f}".format(set_evaluator.specificityMean()))
    print(" - ACC:  {:.4f}".format(set_evaluator.accuracyMean()))
    print(" - AUC:  {:.4f}".format(set_evaluator.rocAucMean()))

    return True

def main(args : argparse.Namespace) -> bool:
    onh_image = None
    onh_vessel_mask = None
    if args.path_image is not None:
        # We load the user requested image
        onh_image = skimage.io.imread(args.path_image)
        onh_vessel_mask = None if args.path_mask is None else skimage.io.imread(args.path_mask)
        if not args.suppress_transform:
            selector = aabb_selector.AABBSelector()
            c1, r1, c2, r2 = selector.select(onh_image)
            onh_image = skimage.util.img_as_float(onh_image[r1 : r2, c1 : c2])
            onh_vessel_mask = skimage.util.img_as_float(onh_vessel_mask[r1 : r2, c1 : c2]) if onh_vessel_mask is not None else None
            onh_image = skimage.transform.resize(onh_image, (256, 256))
            onh_vessel_mask = skimage.transform.resize(onh_vessel_mask, (256, 256))
        return evaluateExample(onh_image, onh_vessel_mask, Path(args.path_image).stem, args)
    if args.path_data is not None:
        # We load a dataset (for either sample or full set evaluation).
        rfi_io = RFIIO()
        rfi_set = rfi_io.load(Path(args.path_data))
        if (args.data_idx is not None) or (args.data_name is not None):
            # We evaluate a single sample from the dataset.
            idx = np.argmax([sample.name == args.data_name for sample in rfi_set]) if args.data_name is not None else args.data_idx
            sample = rfi_set[idx]
            return evaluateExample(sample.image, sample.vessel_mask, sample.name, args)
        else:
            # We evaluate the whole dataset.
            return evaluateDataset(rfi_set, args)
    return False

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    if not main(args):
        print("The evaluation of segmentation was unsuccsessful. Please make sure that you define at least either '--path_image' or '--path_data'")
