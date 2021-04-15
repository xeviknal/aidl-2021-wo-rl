# Car Racing with RL - UPC AIDL winter 2021

This repo contains code and experiments attempting to solve the Car Racing environment from OpenAI's Gym toolkit.

This is the final project of the group under the supervision of Juanjo Nieto for the UPC School's Artificial Intelligence with Deep Learning postgraduate course, online edition of winter 2020-2021. The team members are:
* Juanjo Nieto (Advisor)
* Xavier Canal
* Rubén Martínez
* Alvaro Navas
* Jaime Pedret

## Running the code

1. Clone the repo.
2. Install the dependencies
   1. (Ubuntu 20.04) Run the `install.sh` script. It will install all deb dependencies as well as the Conda environment, and it will create a new virtual environment called `car-racing` in which all the pip dependencies listed in `requirements.txt` will be installed.
   2. Alternatively, follow these steps in your system:
      1. Install the equivalent packages in your system: `build-essential` `wget` `swig` `gcc` `libjpeg-dev` `zlib1g-dev` `xvfb` `python-opengl` `ffmpeg` `xserver-xorg-core` `xorg-x11-server-Xvfb` `htop`
      2. Install Conda from https://www.anaconda.com/products/individual#Downloads
      3. Create a virtual environment with Conda: `conda create --name car-racing python=3.8`
      4. Install the pip requirements: `pip install -r requirements.txt`
 3. Open `main.py` and change the hyperparameters as needed.
 4. Run `python main.py`


## Experiment results

### REINFORCE (Vanilla Policy Gradient)
#### Hypothesis
#### Experiment Setup
#### Results
#### Conclusions

### REINFORCE with Baseline
#### Hypothesis
#### Experiment Setup
#### Results
#### Conclusions

### Proximal Policy Optimization (PPO)
#### Hypothesis
#### Experiment Setup
#### Results
#### Conclusions
