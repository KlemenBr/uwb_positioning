# uwb_positioning


# Requirements
To download and extract the data set 16.4 GB of free disk space is needed. 

All data in data set is already preprocessed so running preprocessing is not needed. You can run it in case you want to reproduce the process or if you want to analyze or to review the process. **In case you want to run the preprocessing scripts, additional 12.5 GB of free disk space is needed: total 28.9 GB of free disk space.**

# Install NVIDIA TensorFlow Docker Image
For Ubuntu 22.04 please follow the instructions on the following link:
https://docs.docker.com/desktop/install/ubuntu/

## Docker image with NVIDIA GPU support
If you have an NVIDIA GPU and you want to run the experiments using GPU acceleration, please follow the following instructions.

To install NVIDIA GPU docker support please follow the instructions on https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

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

docker run -u $(id -u):$(id -g) -it --rm -v $PWD:/tmp -w /tmp tf-cpu /bin/bash




