# Master Thesis

The goal is a multi-person 3D motion forecasting method that takes advantage of person-person interaction in order to refine its estimates.


## Code Requirements

The code was implemented with PyTorch version 1.7 running on CUDA 10.2.

## Structure of Code

The main scripts are *training.py* for training new models, *prediction.py* for running these models and *evaluation.py* to evaluate the performance. The script in *prediction.py* loads an existing model, infers motions on test data and stores the predictions on disk for reproducability. The script in *evaluation.py* loads these predictions, visualizes them and computes performance metrics. Use the help flag, e.g.,

```
$ python training.py -h
```

for info on how to use them. Training can be run on clusters (Vision cluster, Claix18) via bash scripts. To provide non-default hyperparameters create a configuration file with the same fields as *config.py* and provide the corresponding file as a parameter to *training.py*.

The files *baselines.py*, *rnn.py* and *transformer.py* contain code for creating the models. Code related to data loading and representation is located in *conversion.py* and *dataset.py* and visualisation in *visualisation.py*.

The code for preprocessing the datasets (AMASS, PKU-MMD phase 1, NTU-RGB+D 120) is located in the files *preprocess_XXX.py* and in *util.py*. In order to run the preprocessing, change the last lines to:

```
if __name__ == "__main__":
    preprocess_first()
    preprocess_second()
    ...
```

Note that the preprocessing code for PKU-MMD is located in *preprocess_metrabs.py* whereas *preprocess_pkummd.py* contains important code snippets for all 3 preprocessing pipelines. This is a relict from when we used the Kinect poses instead of MeTRAbs from Sárándi et al. Note furthermore that *preprocess_amass.py* expects that forward kinematics have already been computed and the resulting joint locations have been stored as pickle files.

Script files beginning with *test_* contain functions with common use cases in order to verify the correctness of other scripts. You can use them to verify that a change you implemented does not introduce unwanted behaviour.

Lastly, the script *compute_stats.py* computes interesting statistics about the datasets, e.g., the number of frames with 0, 1 or 2 poses visible. The script *tex_evaluate.py* is similar *evaluation.py* but outputs TeX code which produces columns in a table. The script *vis_figues.py* produces plots which are used in the thesis document and the final presentation. Create videos by calling

```
ffmpeg -framerate 25 -start_number 0 -i image_name_%02d.png target.mp4
```
