
import argparse
from pathlib import Path

from rbv_sin.utils import image_splitter, wrappers, inp_data_gen

parser = argparse.ArgumentParser()
parser.add_argument("--cp_seg", default="", type=str, help="Path to the checkpoint of the segmentation network.")
parser.add_argument("--cp_inp", default="", type=str, help="Path to the checkpoint of the inpainting network.")
parser.add_argument("--name_seg", default="", type=str, help="Name (with format) of the resaved segmentation network.")
parser.add_argument("--name_inp", default="", type=str, help="Name (with format) of the resaved inpainting network.")
parser.add_argument("--split_stride", default=32, type=int, help="Stride used during image splitting and composition for vessel segmentation.")
parser.add_argument("--path_save", default=None, type=str, help="Path to the directory where the result should be saved.")

def main(args : argparse.Namespace) -> None:
    """Initialises minimal number of objects for model loading and then saves the weights in the requested location."""
    splitter = image_splitter.ImageSplitter((128, 128), (args.split_stride, args.split_stride))
    data_transformer = inp_data_gen.InpaintDataTransformer()

    seg_wrapper = wrappers.SINSegmentationTileWrapper(args.cp_seg, [128, 128, 3], splitter)
    inp_wrapper = wrappers.SINInpaintingWrapper(args.cp_inp, {"generator" : [256, 256, 4], "vgg19" : [256, 256, 3]}, data_transformer)

    path_save = Path(args.path_save)
    if not Path.exists(path_save):
        Path.mkdir(path_save, parents=True)

    seg_wrapper.network._saveModels(path_save, args.name_seg)
    inp_wrapper.network._saveModels(path_save, args.name_inp)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
