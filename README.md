# FabuLight-ASD

This repository contains the code of [FabuLight-ASD](https://link.springer.com/article/10.1007/s00521-024-10792-0) (**F**ace, **a**udio, and **b**ody **u**tilisation for **Light**weight **A**ctive **S**peaker **D**etection) as well as some saved model weights.

FabuLight-ASD is an active speaker detection model based on [Light-ASD](https://github.com/Junhua-Liao/Light-ASD), which incorporates body pose as an additional modality.

***

## Dependencies

### Required packages

Install the required packages.
```
python3 -m pip install -r requirements.txt
```

### Dataset download and feature extraction

To train the model, it is suggested to download the [WASD dataset](https://tiago-roxo.github.io/WASD/).

Files ***prepare_setup.py*** and ***create_dataset.py*** are responsible for the creation of the dataset folder structure and downloading the WASD dataset, and the subsequent extraction of audio, face crops, and body crops.

Please follow the description provided at the [WASD repository](https://github.com/Tiago-Roxo/WASD#download-dataset) for the details of the procedure.

### Dataset cleaning and update of CSV files

**Note: the steps below are necessary for FabuLight-ASD to correctly utilise body joint information**

After downloading the WASD dataset and extracting the audio of every scene, and the face and body crops of every individual in each frame, with the subsequent organisation of the extracted data in subfolders that characterise the kind of data and the dataset split to which it is assigned, the dataset must be cleaned and the downloaded CSVs properly updated.

There is a tiny inconsistency in the body and face crops of some individuals in the training split of WASD, but which may render some error messages due to the presence of tensors of inconsistent sizes. To fix it, the following face crops should be removed:

* ***0.07.png***, ***29.83.png***, and ***29.90.png*** at subfolder ***WASD/clip_videos/G16p2gY-26M_905-935/G16p2gY-26M_905-935_0000_0060_1***
* ***0.07.png***, ***29.83.png***, and ***29.90.png*** at subfolder ***WASD/clip_videos/G16p2gY-26M_905-935/G16p2gY-26M_905-935_0000_0060_2***
* ***0.07.png*** and ***29.87.png*** at subfolder ***WASD/clip_videos/Ve_wd6pcZwY_70-100/Ve_wd6pcZwY_70-100_0000_0060_1***
* ***0.07.png***, ***26.20.png***, ***26.23.png***, ***26.43.png***, and ***29.87.png*** at subfolder ***WASD/clip_videos/Ve_wd6pcZwY_70-100/Ve_wd6pcZwY_70-100_0000_0060_2***
* ***0.07.png*** at subfolder ***WASD/clip_videos/WZkEgX7lfQs_10-35/WZkEgX7lfQs_10-35_0000_0060_1***
* ***0.07.png*** at subfolder ***WASD/clip_videos/WZkEgX7lfQs_10-35/WZkEgX7lfQs_10-35_0000_0060_2***
* ***0.07.png*** at subfolder ***WASD/clip_videos/WZkEgX7lfQs_10-35/WZkEgX7lfQs_10-35_0000_0060_3***
* ***0.07.png*** at subfolder ***WASD/clip_videos/WZkEgX7lfQs_10-35/WZkEgX7lfQs_10-35_0000_0060_4***
* ***0.07.png*** at subfolder ***WASD/clip_videos/WZkEgX7lfQs_10-35/WZkEgX7lfQs_10-35_0000_0060_5***

The removal of the aforementioned face crops from the ***WASD/clip_videos*** might lead to other error messages during training, which can be fixed by updating ***WASD/csv/train_loader.csv*** and ***WASD/csv/train_loader_body.csv*** with the versions avaiable at [this link](https://www2.informatik.uni-hamburg.de/wtm/corpora/WASD_updated_csvs_with_body_pose_information.tar.gz).

Additionally, extract ***train_orig_kp_body.csv*** and ***val_orig_kp_body.csv*** to the same subfolder (***WASD/csv***). These CSVs contain the body joint coordinate data for every individual in each frame.

***

## Training the model

To train the model, one should execute ***fabulight.py***, which includes a variety of arguments:

* mode: execution mode (train, test, or demo) (***default: test***)
* lr: initial learning rate (***default: 0.001***)
* lrDecay: learning rate decay (***default: 0.95 per epoch***)
* maxEpoch: maximum number of training epochs (***default: 30***)
* testInterval: number of training epochs between evaluation and model saving procedures (***default: 1***)
* batchSize: upper limit of the total number of frames comprising all input samples sent in a single batch (***default: 2000***)
* nDataLoaderThread: number of dataloader workers (***default: 64***)
* optim: optimiser (Adam or AdamW) (***default: Adam***)
* lossScheduling: loss decay scheduling approach (***default: epochStep***)
* numWarmupEpochs: number of warmup epochs for training (***default: 0***)
* initialTemp: initial temperature value (utilised during training) (***default: 1.3***)
* tempDecayType: type of decay employed by the temperature (***default: linear***)
* tempDecayRate: rate of decay of the temperature (***default: 0.02 per epoch***)
* sizeVideoInput: the dimension of the width of the face crops in pixels (***default: 112***)
* bodyPose: whether body pose will be utilised as an additional modality (***default: false***)
* bodyPoseAugProb: probability a body joint is selected for augmentation (***default: 0***)
* bodyPoseAugMag: maximum alteration in the position of the selected body joint (***default: 0***)
* upperBody: whether only the body joints from the upper body would be employed (***default: false***)
* id: a dummy string to differentiate and identify models, especially if there are two or models produced with an identical set of arguments (***default: ""***)

Most model parameters keep the original values employed at [Light-ASD](https://github.com/Junhua-Liao/Light-ASD)'s implementation.

Utilising the set of arguments above, the trained model will be saved at the ***exps*** subfolder, and within it, in a subfolder that uniquely identifies that model based on the arguments.

If one wants to fine-tune or evaluate an existing model, by executing ***fabulight.py*** with the same parameters that produced the model in the first place, the program is capable of retrieving the existing model and fine-tune it (or continue its training) from its last epoch, or to evaluate its performance at the epoch in which it achieved its highest performance metrics.

## Fine-tuning existing models

Among the provided models, there is the model at folder ***exps/exp_0.001_0.95_60_2000_Adam_epochStep_0.0_1.3_linear_0.02_112_F_fabulight_config***. The subfolder name identifies the following arguments of the model:

* initial learning rate: ***0.001***
* learning rate decay: ***0.95***
* maximum number of training epochs: ***60***
* batch size: ***2000***
* optimiser: ***Adam***
* loss decay scheduling approach: ***epochStep***
* number of warmup epochs: ***0.0*** (number of warmup epochs can assume non-integer values in case one wants to stop the warmup mid-epoch)
* initial temperature: ***1.3***
* type of decay of the temperature: ***linear***
* rate of decay of the temperature: ***0.02*** (per epoch)
* face crop width size: ***112*** (pixels)
* body pose as an additional modality: False (indicated as ***F*** in the subfolder name)
* id: ***fabulight***

The ***exp*** at the beginning of the subfolder string and the ***config*** at its end are arbitrary strings present in all subfolders. Their semantics are respectively "experiment" and "configuration" as they represent model configurations from specific experiments that have been conducted.

In ***exps/exp_0.001_0.95_60_2000_Adam_epochStep_0.0_1.3_linear_0.02_112_F_fabulight_config/model***, one can find the model at its highest performing instance (***best_0044.model***), which was obtained at its 27th epoch, and its instance at its last training epoch (***last_0060.model***).

To fine-tune the model, one needs simply to execute the following command:
```
python3 fabulight.py --mode train --maxEpoch 60 --id fabulight
```

Please notice that only the arguments whose values differ from the default ones must be provided.

Please notice as well that perhaps 64 concurrent dataloader workers might be too much depending on the available architecture. Include the argument ***--nDataLoaderThread*** followed by the adequate number of dataloader workers, if necessary.

To simply evaluate the same model at its highest performing epoch, run the same command with ***--mode test*** as an additional argument, as follows:
```
python3 fabulight.py --mode test --maxEpoch 60 --id fabulight
```

To employ the same model at its highest performing checkpoint in a demo utilising the computer's camera, run the same command with ***--mode demo*** as an additional argument, as follows:
```
python3 fabulight.py --mode demo --maxEpoch 60 --id fabulight
```

## Additional available models

### Upper body

In ***exps*** folder, one can also find the subfolder ***exps/exp_0.001_0.95_60_2000_Adam_epochStep_0.0_1.3_linear_0.02_112_T_0.0_0.0_upper_fabulight_config***, which is model presented in the paper which utilises information from the body joints exclusively from the upper body. The subfolder name identifies the following arguments of the model:

* initial learning rate: ***0.001***
* learning rate decay: ***0.95***
* maximum number of training epochs: ***60***
* batch size: ***2000***
* optimiser: ***Adam***
* loss decay scheduling approach: ***epochStep***
* number of warmup epochs: ***0.0***
* initial temperature: ***1.3***
* type of decay of the temperature: ***linear***
* rate of decay of the temperature: ***0.02***
* face crop width size: ***112***
* body pose as an additional modality: True (indicated as ***T*** in the subfolder name)
* probability of body joint being selected for augmentation: ***0.0***
* magnitude of the maximum relative displacement of a body joint due to augmentation: ***0.0***
* Utilisation of solely the upper part of the body of the individuals: True (indicated as ***upper*** in the subfolder name)
* id: ***fabulight***

Since most arguments utilise their default values, the number of required arguments to either fine-tune or evaluate the model is very small.

To fine-tune the model, one needs simply to execute the following command:
```
python3 fabulight.py --mode train --maxEpoch 60 --bodyPose --upperBody --id fabulight
```

To evaluate it at its highest performing epoch:
```
python3 fabulight.py --mode test --maxEpoch 60 --bodyPose --upperBody --id fabulight
```

To employ the model at its highest performing checkpoint in a demo utilising the computer's camera:
```
python3 fabulight.py --mode demo --maxEpoch 60 --bodyPose --upperBody --id fabulight
```

### Whole body

In that folder, one can also find the subfolder ***exps/exp_0.001_0.95_60_2000_Adam_epochStep_0.0_1.3_linear_0.02_112_T_0.0_0.0_whole_fabulight_config***, which is model presented in the paper which utilises information from the body joints from the whole body. The subfolder name identifies the following arguments of the model:

* initial learning rate: ***0.001***
* learning rate decay: ***0.95***
* maximum number of training epochs: ***60***
* batch size: ***2000***
* optimiser: ***Adam***
* loss decay scheduling approach: ***epochStep***
* number of warmup epochs: ***0.0***
* initial temperature: ***1.3***
* type of decay of the temperature: ***linear***
* rate of decay of the temperature: ***0.02***
* face crop width size: ***112***
* body pose as an additional modality: True (indicated as ***T*** in the subfolder name)
* probability of body joint being selected for augmentation: ***0.0***
* magnitude of the maximum relative displacement of a body joint due to augmentation: ***0.0***
* Utilisation of solely the upper part of the body of the individuals: False (indicated as ***whole*** in the subfolder name)
* id: ***fabulight***

Similarly to the previous model, most arguments utilise their default values, thus the number of required arguments to either fine-tune or evaluate the model is very small.

To fine-tune the model, one needs simply to execute the following command:
```
python3 fabulight.py --mode train --maxEpoch 60 --bodyPose --id fabulight
```

To evaluate it at its highest performing epoch:
```
python3 fabulight.py --mode test --maxEpoch 60 --bodyPose --id fabulight
```

To employ the model at its highest performing checkpoint in a demo utilising the computer's camera:
```
python3 fabulight.py --mode demo --maxEpoch 60 --bodyPose --id fabulight
```

***

## Citation

In case you used FabuLight-ASD for your research, please cite the following paper:
```
@article{carneiro2024fabulightasd,
	title = {{FabuLight-ASD}: Unveiling Speech Activity via Body Language},
	author = {Hugo Carneiro and Stefan Wermter},
	journal = {Neural Computing and Applications},
	year = {2024},
	doi = {10.1007/s00521-024-10792-0}
}
```
