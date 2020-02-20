# Framewise Polyphonic Transcription 2016
- this repository contains code to reproduce results from a paper about framewise polyphonic piano transcription (https://arxiv.org/abs/1612.05153)

- the code is ported from theano+lasagne to pytorch 1.0.0

- obtain the MAPS dataset (http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/)

- clone this repository

- you'll need python >= 3.5, and a CUDA installation if you want to use a GPU

- you don't need to install anything from within this repo, but you'll need to install some required python packages
```
$ pip install -r requirements_00.txt
$ pip install -r requirements_01.txt
```

## Data and Split Preparation
- to keep things organized, we'd recommend creating these directories:
```
$ mkdir data
$ mkdir splits
$ mkdir runs
```

- link in the MAPS dataset into `data`:
```
$ cd data
$ ln -s <path-to-where-MAPS-was extracted to> .
```

- you'll then need to generate the (partially musically overlapping) splits (called `Configuration II` in the paper), by pointing the helper script to the folder that has the MAPS data
```
$ python create-configuration-II-splits.py data/<MAPS-install-directory>/data
```

- this will create a directory `configuration-II`, containing three textfiles `train`, `valid` and `test`, which contain (audiofile, midifile) pairs. you could then move this subdirectory into `splits`:
```
$ mv configuration-II splits
```

## Training
- the following call will start training the VGG-style network, and keep track of progress in `runs/<vgg-result-directory>`
```
$ python train.py splits/configuration-II/ runs/<vgg-result-directory> VGGNet2016
```

- the following call will start training the all-convolutional network, and keep track of progress in `runs/<allconv-result-directory>`
```
$ python train.py splits/configuration-II/ runs/<allconv-result-directory> AllConv2016
```

- calling `train.py` for the first time will take a while, as it has to compute all spectrogram-label pairs from the audiofile and the midifile. the results of these computations are cached via `joblib` in a folder named `joblib_cache`

- you can track training progress visually by using tensorboard
```
$ tensorboard --logdir runs/<result-directory>
```

## Evaluation
- you can evaluate a VGGNet2016 network-state by using
```
$ python evaluate.py VGGNet2016 runs/<vgg-result-directory>/best_valid_loss_net_state.pkl splits/configuration-II/test
```

- you can evaluate an AllConv2016 network-state by using
```
$ python evaluate.py AllConv2016 runs/<allconv-result-directory>/best_valid_loss_net_state.pkl splits/configuration-II/test
```

- if you evaluate for the first time, the spectrogram-label pairs for the test set need to be computed. the results of these computations are cached via `joblib` in a folder named `joblib_cache`

- you can choose the amount of frames to evaluate on via `start_end`
```
$ python evaluate.py runs/<result-directory>/best_valid_loss_net_state.pkl \
         splits/configuration-II/test \
         --start_end "<start>,<end>"
```

- if you enter anything other than `"<number>,<number>"` for --start_end, the network is evaluated on ALL the data

- there are three network states being tracked, one that tracks the best validation loss, the best validation f-measure, and the current network state.

- with the hyperparameters chosen as close as possible to those used for the original code in theano+lasagne, you should see approximately the following results after ~140 epochs:

- Results on the **first 30[s]** of the pieces in the `Configuration II` test set:
  (this is **the same** evaluation protocol as in the paper from 2016 (as well as in Sigtia'16)!)

| Model        | Precision | Recall | F-Measure |
| ------------ | ---------:| ------:| ---------:|
| VGGnet2016   |  0.8045   | 0.7084 |  0.7476   |
| AllConv2016  |  0.8255   | 0.6399 |  0.7142   |


- Results on the **full length** of the pieces in the `Configuration II` test set:
  (this is **different** from the evaluation protocol in the paper from 2016!)

| Model        | Precision | Recall | F-Measure |
| ------------ | ---------:| ------:| ---------:|
| VGGnet2016   | 0.7806    | 0.6664 |  0.7139   |
| AllConv2016  | 0.8060    | 0.6036 |  0.6854   |


## Scheduler Behavior
- if your learnrate schedules are wonky, please execute
```
$ python check_cheduler.py
```
- the result should look something like this:
![expected result](./check_cheduler.png)


## Credit Assignment (pun wholeheartedly intended)
If you use stuff from this repository, please cite:
```
@inproceedings{kelz_etal_2016
  author    = {Rainer Kelz and
               Matthias Dorfer and
               Filip Korzeniowski and
               Sebastian B{\"{o}}ck and
               Andreas Arzt and
               Gerhard Widmer},
  title     = {On the Potential of Simple Framewise Approaches to Piano Transcription},
  booktitle = {Proceedings of the 17th International Society for Music Information
               Retrieval Conference, {ISMIR} 2016, New York City, United States,
               August 7-11, 2016},
  pages     = {475--481},
  year      = {2016}
}
```
