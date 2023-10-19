# Haptic Material Dataset and Kinematic-motivated Haptic Material Classification Network
An official implementation of the paper _HapNet: A Learning-Based Haptic-Kinematic Model for Surface Material Classification in Robot Perception_, including the HapMat dataset.zip as well as the HapNet .h5 pretrained model checkpoint, along with related files. 

**The paper is now accepted by IEEE CASE 2023!** The open-access version of the paper can be downloaded from [Google Drive](https://drive.google.com/file/d/1221HU043OiFagdT_EtBUv9MBsV-KjA7t/view?usp=sharing). **Feel free to clone our full project, test, and even make improvements on your own.** We also STRONGLY suggest you utilize our given datasets for any testing purpose, especially making it much easier for you to try our pretrained model.

### Abstract
Abstract— The advancement of modern robotic systems is inseparable from the development of artificial sensory modules and functions that provide robots with human-like perception. Among these, haptic sensation is of vital importance for robot interaction with the environment. Many research works have been conducted in the past decades to enable robots to sense the environment, especially in the area of surface material classification. This paper offers a learning-based low-cost but high computational efficient method adopting multiple kinematic modalities, namely the haptic-kinematic data. We introduce a haptic surface material dataset and propose a kinematics-motivated haptic surface material classification network (HapNet) trained and tested on our dataset. The results demonstrate the feasibility and robustness of a purely kinematic haptic scheme without the need for other perceptual modalities.

![image](https://github.com/henryyantq/haptic-kinematics/assets/20149275/5e1b5fc4-1b5d-4130-baa6-f13e0e973492)

### Overall structure of HapNet
**Please refer to our paper for more detailed description.**

![image](https://github.com/henryyantq/haptic-kinematics/assets/20149275/e1944de4-6c9f-49c5-93c2-dd7b0e33fbf9)

### How to run our codes
The core of our project, HapNet, is written and trained in TensorFlow 2. Please follow the instructions below for users of different operating systems. These codes are recommended to be run on GPU-available devices, but notice that they are also made efficient to be deployed on **CPU-only** devices!

#### 1. For Mac users
If you are a Mac user, please follow the [official document by Apple](https://developer.apple.com/metal/tensorflow-plugin/) to install the GPU-enabled version of TensorFlow. You can skip this for a CPU-only implementation.

#### 2. For Windows/Linux users
If you require an **Nvidia** GPU-accelerated implementation, install the Nvidia Driver and CUDA first before you ```pip install -r requirements.txt```. If you wish to run the codes with GPUs from other brands, e.g. AMD, Intel, please follow [the official document by Microsoft](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-tensorflow-plugin) to deploy TensorFlow for DirectML, **which has NOT yet been examined by us**.

#### 3. Train with HapMat dataset
Simply run ```python train.py``` or ```python3 train.py``` will begin the training phase. The ```train.py``` programme automatically preprocesses the data under the ```HapMat dataset``` folder and feeds them to the predefined HapNet model. 

#### 4. Making your own dataset
It is extremely EASY to establish your own haptic-kinematic dataset, but before the retrieval, you need an IMU module (_with at least a 3-axis accelerometer and a 3-axis gyroscope_) **tightly sticked to a rigid object (e.g. a wooden stick)** and connected to a computing device. The computing device should be able to read and save your IMU outputs in **real-time**, and you should be able to know the **operating frequency** (the max. val. and the min. val.) of your IMU module so that the time stamps are correctly assigned for a single read. When you collect the haptic-kinematic data of any surface, all you need is to swipe the rigid object tightened with the IMU across the target surface in a random pattern and at a normal speed (NOT too fast or too slow), and in the meantime, record the 3-axis linear acceleration data (accelerometer) as well as the 3-axis angular acceleration data (gyroscope) with your computing device. The data collection phase should NOT last too short. After the recording, your data should be saved locally on your computing device. The ```preprocess``` function in ```train.py``` and ```infer.py``` natively accepts **.csv file** as the haptic-kinematic data of a single surface material. Therefore, it'll be a lot convenient of you to format your retrieved IMU data as shown in the image below saved it or re-saved it in .csv format:

![image](https://github.com/henryyantq/haptic-kinematics/assets/20149275/9c900a3d-b798-4195-a640-3c0980b9eb7a)

You can read the code directly for more details, since we are making our code as straightforward-to-read as possible.
