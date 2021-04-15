# Car Racing with RL - UPC AIDL winter 2021

This is the repo for the final project of the group under the supervision of Juanjo Nieto for the UPC School's Artificial Intelligence with Deep Learning postgraduate course, online edition of winter 2020-2021. The team members are:
* Juanjo Nieto (Advisor)
* Xavier Canal
* Rubén Martínez
* Alvaro Navas
* Jaime Pedret

The original goal of the project was to train a self-driving model that would allow us to have a vehicle of some sort drive itself through a circuit. One of our members, Rubén Martínez, owns a robot which is well suited for this task, but since the other members did not have easy access to it, it was decided that we would start by training a model in a OpenAI Gym environment which could be later be adapted to the robot, and the tasks were divided so that Rubén would work on the robot side and the rest would work on the Gym side.

In the end the original goal was too ambitious and the project ended up divided in 2 separate parts: the Gym part and the Robot part. This repo contains the Gym part; you may check out the robot part by visiting Rubén's repo at https://github.com/eldarsilver/DQN_Pytorch_ROS .

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

## Reinforcement Learning and Car Racing

Reinforcement Learning (RL) is a *computational approach to goal-directed learning form interaction that does not rely on expert supervision* [(quote)](https://mitpress.mit.edu/books/reinforcement-learning-second-edition). In other words, it's the branch of Machine Learning that tries to achieve a task by using an active agent that reads data from the environment and a "teacher" that gives an extrinsic reward to the model in order to teach it when it's doing well. The agent gets a ***state*** from the environment and performs an ***action*** on the environment, which is then either rewarded, punished or ignored; then the agent gets a new state and the cycle repeats.

![01_intro_pdf](https://user-images.githubusercontent.com/1465235/114918938-f5390500-9e27-11eb-876d-00c59f9d747b.jpg)

OpenAI Gym is a popular framework for training Reinforcement Learning models. Gym provides a set of pre-made environments that allows students and researchers to test different approaches to solve the tasks proposed by each particular environment. One of these environments is [Car Racing](https://gym.openai.com/envs/CarRacing-v0/), which provides an 8-bit-videogame-like environment in which a car appears on a randomly generated circuit. The task to be achieved in this environment is to teach the car to drive itself in order to finish a lap.

![Apr-15-2021 20-10-16](https://user-images.githubusercontent.com/1465235/114917836-b35b8f00-9e26-11eb-9f53-72ba1f8e770f.gif)

The Car Racing environment outputs a ***state*** consisting on a 96x96 RGB image that displays the car and the track from a top-down view, as well as an additional black bar at the bottom of the image which contains various car sensor info. The environment expects an ***action*** input which consists of an array with 3 floating point numbers which represent turning direction (from -1 to 1, representing left to right), throttle (from 0 to 1, representing no throttle to full throttle) and brake (from 0 to 1 too, representing no brake to full brakes) inputs for the car to take. After receiving the action, the environment will return a ***reward*** as well as a new ***state*** consisting of an updated image that reflects the updated position of the car in the track. The environment will also output a *done* boolean value that will be `True` when the car finishes a lap or when it drives outside of the boundaries of the environment.

The default reward is a floating point value that may be positive or negative depending on the performance of the car on the track. The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles visited in the track. For example, if you have finished in 732 frames, the reward will be 1000 - 0.1 * 732 = 926.8 points. The task is considered finished when the agent consistently gets more than 900 points, but the definition of "consistently" is undefined by the environment and it's left for the developer to define it when implementing a solution.

## Our approach

We decided to initially approach the task by using policy-based RL methods, starting from REINFORCE-Vanilla Policy Gradient and implementing more sophisticated algorithms as we understood the behavior, advantages and shortcomings of each algorithm. Our chosen library was PyTorch due to our familiriaty with it and its ease of use.

Before implementing any algorithm, however, we knew from our classes and from additional research that using the vanilla environment as-is would be inefficient. The OpenAI Gym framework allows the use of *wrappers*, pieces of code that "wrap" the default environment in order to alter the outputs to make them more convenient for our purposes. Gym already provides some wrappers but we also implemented a few. The wrappers we ended up using were:
* Monitor: one of Gym's provided wrappers. It provides an easy way to record the output of the environment to a video file.
* GrayScaleObservation: another provided wrapper; it transforms RGB images to monochrome. Useful for reducing dimensionality in cases where the additional RGB info does not provide useful info compared to black and white images.
* FrameStack: another provided wrapper; FrameStack allows us to "stack" frames (states) in order to create a "mini-batch" of sorts for more efficient training.
* FrameSkipper: an original wrapper; FrameSkipper is a companion to FrameStack and allows us to skip stacked frames so that we do not use redundant frames.
* EarlyStop: an original wrapper; when used, the environment will output `done = True` in additional circunstances besides the default ones, such as getting a negative average reward. This allows us to stop the execution early and train with more episodes.

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
