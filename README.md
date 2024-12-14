
# RBV-SIN: Retinal blood vessel segmentation and inpainting networks with multi-level self-attention

This repository contains the implementation of the paper ***Retinal blood vessel segmentation and inpainting networks with multi-level self-attention*** available at [TBD]()

## Licence notice

We need to distinguish between two licences applied to the content of this repository. Simply put, figures showing human retina or blood vessel masks, and neural network weights fall under **Creative Commons Attribution 4.0 International Public License(CC BY 4.0)**. Everything else, i.e. the code and all other files in the repository fall under the **MIT** licence.

It is not entirely clear, whether trained network weights are considered a derivative work of the dataset used to train the network. Nevertheless, we release the weights under the same licence under which we obtained the source dataset, that is **CC BY 4.0** ([https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)). The legal codes of both licences are available in the root of this repository where *LICENSE* contains the notice for **MIT** and *LICENSE_CC4* for **CC BY 4.0**.

We adapted the FIVES retinal vessel segmentation dataset ([https://doi.org/10.1038/s41597-022-01564-3](https://doi.org/10.1038/s41597-022-01564-3)) by K. Jin et al. to train our networks and create our figures.

## Python setup

The project was fully developed in **Python 3.10.11** with **Tensorflow** version **2.10.0** for neural network optimisation. We used the standard packages written in the following list:

- numpy
- matplotlib
- scikit-learn
- scikit-image
- opencv-python
- opencv-contrib-python
- tensorflow

The file *requirements.txt* contains a list of these packages for easier installation with pip. In addition, we provide *requirements_frozen.txt*, which contains the exact versions of all packages installed in our project environment during verification of the code in this repository.

## Data preprocessing

Trying out our method on images using the testing scripts doesn't require specific data preprocessing, however, if one wants to train using our code or evaluate on an entire set of images at once then it is necessary to prepare the data in the expected format. This can be done with the script **data_process.py**. The script has the following arguments:

- `--dataset` Specifies which dataset should be processed or whether we want to split and existing dataset into patches for vessel segmentation. The supported values are `fives`, `chase`, `patch`
- `--path_data` Directory of the [FIVES](https://doi.org/10.1038/s41597-022-01564-3) or [CHASE_DB1](https://researchdata.kingston.ac.uk/96/) dataset as downloaded from primary sources. For `--dataset=patch`, this should be the path directly to our dataset file (i.e. one of the files created through `--dataset=fives` or `--dataset=chase`)
- `--path_roi` Path to a directory with region of interest bounding boxes. It should be the *rois* directory in this repository.
- `--path_save` Path to a directory where the script saves the generated data files.

Other arguments are less important and can be looked up in the script itself. Let us consider the directory structure to be the following tree for the rest of the examples described in this file. All scripts should be executed from the root directory of this repository.

```
.
+-- CHASE # This is the root CHASE directory
+-- FIVES # This is the root FIVES directory
|   +-- test
|   +-- train
|   +-- Quality Assessment.xlsx
+-- [This repository]
|   +-- data
|   +-- nn
|   |   +-- seg_20240212_e0100
|   |   +-- inp_20240213_e0100
|   +-- rbv_sin
|   +-- results
|   +-- rois
|   +-- .gitignore
|   +-- ...
```

With the directory tree above, we can generate CHASE and FIVES data in the required format with the following commands.

```
cd path/to/this/repository
python data_process.py --dataset=chase --path_data=../CHASE --path_roi=rois --path_save=data
python data_process.py --dataset=fives --path_data=../FIVES --path_roi=rois --path_save=data
```

And to generate data for blood vessel segmentation training assuming the commands above finished successfully.

```
cd path/to/this/repository
python data_process.py --dataset=patch --path_data=data/chase_train_256.npz --path_save=data --patch_stride=32
python data_process.py --dataset=patch --path_data=data/fives_train_256.npz --path_save=data --patch_stride=64
```

## Network weights

We provide our trained network weights in the *Releases* section of GitHub. The directory **nn**, contained within the release file, should be placed in the location indicated by the directory tree above.

## Inpainting

### Testing

The inpainting network can be tested using **test_inp.py** script. We can test the network on preprocessed data or directly on full retinal images.

Testing on a full retinal image will prompt a window where the user has to select a region of interest, preferably with the ONH, and this region will be cropped, resized and evaluated. *The application works by drawing a rectangle with left click drag (it is possible to redraw the rectangle) and confirming the selection with a right click.*

```
python test_inp.py --path_image=../FIVES/test/Original/4_A.png --path_mask="../FIVES/test/Ground truth/4_A.png" --cp_seg=nn/seg_20240212_e0100 --cp_inp=nn/inp_20240213_e0100 --path_save=results/inp_test --show_results
```

Testing on preprocessed data does not require RoI selection so the sample will be evaluated directly.

```
python test_inp.py --path_data=data/fives_test_256.npz --data_name=4_A --cp_seg=nn/seg_20240212_e0100 --cp_inp=nn/inp_20240213_e0100 --path_save=results/inp_test --show_results
```

It is also possible to evaluate inpainting on an entire preprocessed dataset, which will save the results in the given directory.

```
python test_inp.py --path_data=data/chase_test_256.npz --cp_seg=nn/seg_20240212_e0100 --cp_inp=nn/inp_20240213_e0100 --path_save=results/inp_test/chase_test
```

### Training

The inpainting network can be trained using **train_inp.py** script. Training requires the preprocessed dataset. To train with validation, use argument `--validation` with the fraction of data used for validation, e.g. `0.2`. Additional training parameters such as the learning rate can be looked up within the script.

```
python train_inp.py --path_train=results/inp_train --path_data=data/fives_train_256.npz --epochs=100 --cp_epoch_delta=10
```

The training can be continued from a particular checkpoint by using the `--continuation` argument as written in the following command.

```
python train_inp.py --path_train=results/inp_train --path_data=data/fives_train_256.npz --epochs=100 --cp_epoch_delta=10 --continuation=cp_name
```

Where `cp_name` is the checkpoint directory name and has the form like `inp_20240219_e0006_l0.007680` or `inp_20240219_e0006`. Note that training runs with and without validation should not be combined through continuation because the model expects all losses and metrics defined at the first epoch to be present in all epochs.

## Segmentation

### Testing

The segmentation network can be tested using **test_seg.py** script. We can test the network on preprocessed data or directly on full retinal images.

Testing on a full retinal image will prompt a window where the user has to select a region of interest, preferably with the ONH, and this region will be cropped, resized and evaluated. *The application works by drawing a rectangle with left click drag (it is possible to redraw the rectangle) and confirming the selection with a right click.*

```
python test_seg.py --path_image=../FIVES/test/Original/4_A.png --path_mask="../FIVES/test/Ground truth/4_A.png" --cp_seg=nn/seg_20240212_e0100 --path_save=results/seg_test --show_results
```

Testing on preprocessed data does not require RoI selection so the sample will be evaluated directly.

```
python test_seg.py --path_data=data/fives_test_256.npz --data_name=4_A --cp_seg=nn/seg_20240212_e0100 --path_save=results/seg_test --show_results
```

It is also possible to evaluate inpainting on an entire preprocessed dataset, which will save the results in the given directory.

```
python test_seg.py --path_data=data/chase_test_256.npz --cp_seg=nn/seg_20240212_e0100 --path_save=results/seg_test/chase_test
```

### Training

The inpainting network can be trained using **train_seg.py** script. Training requires the preprocessed dataset. To train with validation, use argument `--validation` with the fraction of data used for validation, e.g. `0.2`. Additional training parameters such as the learning rate can be looked up within the script.

```
python train_seg.py --path_train=results/seg_train --path_data=data/fives_train_256_patches_128_64.npz --epochs=100 --cp_epoch_delta=50
```

The training can be continued from a particular checkpoint by using the `--continuation` argument as written in with the following command.

```
python train_seg.py --path_train=results/seg_train --path_data=data/fives_train_256_patches_128_64.npz --epochs=100 --cp_epoch_delta=10 --continuation=cp_name
```

Where `cp_name` is the checkpoint directory name and has the form like `cp_name=seg_20240219_e0040_l0.735168` or `cp_name=seg_20240219_e0040`. Note that training runs with and without validation should not be combined through continuation because the model expects all losses and metrics defined at the first epoch to be present in all epochs.
