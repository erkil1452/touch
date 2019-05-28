# Learning the signatures of the human grasp using a scalable tactile glove

## Introduction
This is a Pytorch based code for object classification and object estimation methods presented in the paper "Learning the signatures of the human grasp using a scalable tactile glove".

It relies on Pytorch 0.4.1 (or newer) and the dataset that can be downloaded separately from [http://humangrap.io](http://humangrap.io) .


## System requirements

Requires CUDA and Python 3.6+ with following packages (exact version may not be necessary):

* numpy (1.15.4)
* torch (0.4.1)
* torchfile (0.1.0)
* torchvision (0.2.1)
* scipy (1.1.0)
* scikit-learn (0.19.1)

## Dataset preparation

1. Download the `classification` and/or `weights` dataset from [http://humangrap.io](http://humangrap.io) .
2. Extract the dataset metadata.mat files to a sub-folder `data\[task]`. The resulting structure should be something like this:
```
data
|--classification
|    |--metadata.mat
|--weights
        |--metadata.mat
```
The images in the dataset are for illustration only and are not used by this code. More information about the dataset structure is availble in [http://humangrap.io](http://humangrap.io) .

3. Alternatively, extract the dataset to a different folder and use a runtime argument `--dataset [path to metadata.mat]` to specify its location.

## Object classification

Run the code from the root working directory (the one containing this readme).

### Training
You can train a model from scratch for `N` input frames using:
```
python classification/main.py --reset --nframes N
```
You can change the location of the saved snapshots using `--snapshotDir YOUR_PATH`.

### Testing
You can test the provided pretrained model using:
```
python classification/main.py --test --nframes N
```

## History
Any necessary changes to the dataset will be documented here.

* **May 2019**: Original code released.

## Terms
Usage of this dataset (including all data, models, and code) is subject to the associated license, found in [LICENSE](http://humangrasp.io/license.html). The license permits the use of released code, dataset and models for research purposes only.

We also ask that you cite the associated paper if you make use of this dataset; following is the BibTeX entry:

```
@article{
	SSundaram:2019:STAG,
	author = {Sundaram, Subramanian and Kellnhofer, Petr and Li, Yunzhu and Zhu, Jun-Yan and Torralba, Antonio and Matusik, Wojciech},
	title = {Learning the signatures of the human grasp using a scalable tactile glove},
	journal={Nature},
	volume={569},
	number={7758},
	year={2019},
	publisher={Nature Publishing Group}
	doi = {10.1038/s41586-019-1234-z}
}
```




## Contact

Please email any questions or comments to [info@humangrasp.io](mailto:info@humangrasp.io).
