The code contains the preprocessing scripts and experiments that work on [UWB Positioning and Tracking Data Set](https://zenodo.org/record/7629141). The code demonstrate the UWP positioning technique with ranging error mitigation using deep learning-based ranging error estimation by convolutional neural networks (CNN) using TensorFlow deep learning platform.

# Published Results
This repository contains the code that was used to produce the results published in the following scientific paper:
Bregar, K. Indoor UWB Positioning and Position Tracking Data Set. Sci Data 10, 744 (2023). [https://doi.org/10.1038/s41597-023-02639-5](https://doi.org/10.1038/s41597-023-02639-5)

# Requirements
The code is written and tested on a computer with **Ubuntu 22.04** Linux OS distribution. It should be possible to replicate the experiments on any computer with one of major Linux OS distributions that support Docker. It should also be possible to run the experiments on Windows computers with Docker installed but **any of those configurations wasn't tested**.

To download and extract the data set 16.4 GB of free disk space is needed. 

All data in data set is already preprocessed so running preprocessing is not needed. You can run it in case you want to reproduce the process or if you want to analyze or to review the process. **In case you want to run the preprocessing scripts, additional 12.5 GB of free disk space is needed: total 28.9 GB of free disk space.**

It is adwised to have a computer with 32 GB of RAM.

# Install NVIDIA TensorFlow Docker Image
For Ubuntu 22.04 please follow the instructions on the following link:
[https://docs.docker.com/desktop/install/ubuntu/](https://docs.docker.com/desktop/install/ubuntu/)

## Docker image with NVIDIA GPU support
If you have an NVIDIA GPU and you want to run the experiments using GPU acceleration, please follow the following instructions.

To install NVIDIA GPU docker support please follow the instructions on [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

Build the nvidia-tf docker image

```
docker build -f ./docker/nvidia-tf-gpu -t nvidia-tf .
```

To run the NVIDIA TensorFlow Docker image with GPU support in terminal on your local files run the following command:

```
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $PWD:/tmp -w /tmp nvidia-tf /bin/bash
```


## Docker image with CPU support
If you don't have an NVIDIA GPU or you just don't want to use GPU for the experiments, you can follow the following instructions.

Build CPU image:
```
docker build -f ./docker/tf-cpu -t tf-cpu .
```
To run TensorFlow without CPU support:
```
docker run -u $(id -u):$(id -g) -it --rm -v $PWD:/tmp -w /tmp tf-cpu /bin/bash
```

# Running the Examples
## Start Docker Container Inside Repository
Docker with NVIDIA GPU support
```
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $PWD:/tmp -w /tmp nvidia-tf /bin/bash
```

Docker with CPU support
```
docker run -u $(id -u):$(id -g) -it --rm -v $PWD:/tmp -w /tmp tf-cpu /bin/bash
```

## Download data set
First, the **UWB Positioning and Tracking Data Set** has to be downloaded. You can do that by running the cript **download.sh**. Grab a cup of coffe and wait. It can easily take 10 minutes to download the data set. 
```
bash preprocess.sh
```
As mentioned before, to download and extract the data set 16.4 GB of free disk space is needed. If you want to do complete preprocessing (already done but if you need it to review the process), additional 12.5 GB of free disk space is needed (**total of 28.9 GB**)!


## Run Experiments
When you have the running docker container inside the root **uwb_positioning** path, move to the folder technical_validation or preprocess (depends on what you want actions you want to recreate or review). If you want to review the actual positioning and data evaluation processes, change the directory to the **technical_validation**.

Results from all experiments are already collected in a data set in folder **data_set/technical_validation** but all experiments are there for the sake of reproducability of results.

```
cd technical_validation
```

The deep learning model for estimating the ranging error is being trained with python script **train_ranging_error.py**. The ranging error estimation models are already a part of this repository in folder **technical_validation**. Run the script only if you want to recreate models for the sake of reproducability of results. The process takes approximatelly 2 hours on an NVIDIA GTX1650 GPU and probably takes 5 to 10-times more on an average modern 6-core Intel CPU without NVIDIA GPU acceleration.

```
python3 train_ranging_error.py
```

The list of other experiments:
- cir_min_max_mean.py
- los_nlos.py
- range.py
- range_error.py
- range_error_histograms.py
- range_error_histograms_loc2_loc3.py
- positioning.py
- rss.py

Experiment **positioning.py** is the main experiment which demonstrates the use of ranging error estimates to improve the accuracy of indoor positioning.

```
python3 positioning.py
```


# Authors and License

Author of data set in this repository is Klemen Bregar, klemen.bregar@ijs.si. Copyright (C) 2023 SensorLab, Jožef Stefan Institute, [sensorlab@ijs.si](https://sensorlab.ijs.si/).

This work is licensed under the Apache License 2.0.a License.

# Funding

The research leading to these results has received funding from the European Horizon 2020 Programme project eWINE under grant agreement No. 688116.



