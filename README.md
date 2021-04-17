# Car Racing with RL - UPC AIDL winter 2021

This is the repo for the final project of the group under the supervision of Juanjo Nieto for the UPC School's Artificial Intelligence with Deep Learning postgraduate course, online edition of winter 2020-2021. The team members are:
* Juanjo Nieto (Advisor)
* Xavier Canal
* Rubén Martínez
* Alvaro Navas
* Jaime Pedret

The original goal of the project was to train a self-driving model that would allow us to have a vehicle of some sort drive itself through a circuit. One of our members, Rubén Martínez, owns a robot which is well suited for this task, but since the other members did not have easy access to it, it was decided that we would start by training a model in a OpenAI Gym environment which could be later be adapted to the robot, and the tasks were divided so that Rubén would work on the robot side and the rest would work on the Gym side.

In the end the original goal was too ambitious and the project ended up divided in 2 separate parts: the Gym part and the Robot part. This repo contains the Gym part; you may check out the robot part by visiting Rubén's repo at https://github.com/eldarsilver/DQN_Pytorch_ROS .

## Running the code **(NEEDS TO BE REDONE)**

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

![RL basic diagram](/readme_media/RL.jpg)

OpenAI Gym is a popular framework for training Reinforcement Learning models. Gym provides a set of pre-made environments that allows students and researchers to test different approaches to solve the tasks proposed by each particular environment. One of these environments is [Car Racing](https://gym.openai.com/envs/CarRacing-v0/), which provides an 8-bit-videogame-like environment in which a car appears on a randomly generated circuit. The task to be achieved in this environment is to teach the car to drive itself in order to finish a lap.

![Agent choosing random actions](/readme_media/randomagent.gif)

The Car Racing environment outputs a ***state*** consisting on a 96x96 RGB image that displays the car and the track from a top-down view, as well as an additional black bar at the bottom of the image which contains various car sensor info. The environment expects an ***action*** input which consists of an array with 3 floating point numbers which represent turning direction (from -1 to 1, representing left to right), throttle (from 0 to 1, representing no throttle to full throttle) and brake (from 0 to 1 too, representing no brake to full brakes) inputs for the car to take. After receiving the action, the environment will return a ***reward*** as well as a new ***state*** consisting of an updated image that reflects the updated position of the car in the track. The environment will also output a *done* boolean value that will be `True` when the car finishes a lap or when it drives outside of the boundaries of the environment.

The default reward is a floating point value that may be positive or negative depending on the performance of the car on the track. The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles visited in the track. For example, if you have finished the track in 732 frames, the reward will be 1000 - 0.1 * 732 = 926.8 points. The task is considered finished when the agent consistently gets more than 900 points, but the definition of "consistently" is undefined by the environment and it's left for the developer to define it when implementing a solution.

The Car Racing environment features a physics model that affects the behaviour of the car. The car has rear-wheel propulsion and has very high acceleration, which makes it very easy for the car to oversteer and drift, making it very hard for a human player to regain control once the car starts skidding.

## First steps

We decided to initially approach the task by using policy-based RL methods, starting from REINFORCE-Vanilla Policy Gradient and implementing more sophisticated algorithms as we understood the behavior, advantages and shortcomings of each algorithm. Our chosen library was PyTorch due to our familiriaty with it and its ease of use.

Before implementing any algorithm, however, we knew from our classes and from additional research that using the vanilla environment as-is would be inefficient. The OpenAI Gym framework allows the use of *wrappers*, pieces of code that "wrap" the default environment in order to alter the outputs to make them more convenient for our purposes. Gym already provides some wrappers but we also implemented a few. The wrappers we ended up using were:
* Monitor: one of Gym's provided wrappers. It provides an easy way to record the output of the environment to a video file.
* GrayScaleObservation: another provided wrapper; it transforms RGB images to monochrome. Useful for reducing dimensionality in cases where the additional RGB info does not provide useful info compared to black and white images.
* FrameStack: another provided wrapper; FrameStack allows us to "stack" frames (states) in order to create a "mini-batch" of sorts for more efficient training.
* FrameSkipper: an original wrapper; FrameSkipper allows us to "skip" frames: instead of choosing an action for each frame, we use the same action for all of the skipped frames. This allows us to reduce the amount of actions we need to calculate.
* EarlyStop: an original wrapper; when used, the environment will output `done = True` in additional circunstances besides the default ones, such as getting a negative average reward. This allows us to stop the execution early and train with more episodes.

***NEEDS REVIEW*** There is also the issue of defining what "consistently" means when trying to get a reward of "consistently more than 900 points". We settled on calculating a *running reward* that accumulates the previously obtained rewards and calculates an average of sorts that represents the reward you can expect from the model at a specific stage of training, which is calculated as `running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward`, where `episode_reward` is the reward we obtain for each lap attempt.

For all of our experiments, the chosen optimizer was Adam, since it seems to be the default optimizer for pretty much any task. We did not experiment with additional optimizers.

In order to simplify the code, we decided to have a discrete set of actions initially. The Car Racing environment accepts floating point values as input, which makes it possible to finetune the car driving experience by having contiguous action values, but discrete separate actions allows us to understand the environment better by letting us experiment with the action values and simplifies the code by not having to calculate the additional probability distributions needed to calculate the contiguous values. Our goal was to code contiguous values as a later feature in order to compare experiments but unfortunately we did not have the time to do so.

## Deep neural network

The next step is defining a deep neural network architecture that can process the state.

As previously said, the Car Racing environment outputs a state that consists of a 96x96 pixel RGB image. The obvious choice then is to use some sort of model based on convolutional layers to extract the image features and work with them.

The state is a very small image composed with very simple graphics with flat colors, so a complex network with dozens of layers isn't needed. We designed a very simple model with 3 convolutional layers intercalated with 3 pooling layers and 3 fully connected layers at the end.

![Our model](/readme_media/policy.png)

We experimented with different network configurations and we ended up with slight differences for each implemented RL method, but the basic structure shared among them is the one described above.

## Policy-based RL

There are many methods to do Reinforcement Learning, which can all be classified using different criteria. One of the most common criteria is whether a method is *value-based* or *policy-based*.

A policy-based method maps the observations (*states*) that the agent obtains and maps them to *actions*, whereas a value-based method will output a measure of quality of each possible action. For example, if we're using a map application to find a route between 2 places, a policy-based method would output the best possible route, and a value-based method will output a value associated to all possible routes, such as estimated travel time.

The Car Racing environment and task are very simple: the possible actions that the car may take are limited and there are no instances of having to decide between multiple paths or any other complex scenarios which may need the model to decide between 2 equally valid actions. Thus, a policy-based method seems better suited for this task.

For policy-based methods, there are 2 important functions:
* The **Return G_t** function is a function that calculates the future reward starting from a timestep *t*. It can be thought of as the sum of all rewards starting from timestep *t+1* up to a final timestep *T*. In scenarios where it's impossible to know what the value of the final timestep *T* will be, a *discount rate ɣ* is applied to each additional timestep, so that the final value of *G_t* will always be finite.
* The **Objective J(θ)** function returns the expected value of *G_t* given a set of parameters *θ* from our *policy* (our deep neural network).

By estimating the gradient of *J(θ)* with respect to each policy parameter, we can then use stochastic gradient ascend and backpropagation to update the parameters and train the policy, like this:

![](/readme_media/RLformula.jpg)

Where *ɑ* is the *learning rate* that we use to regulate the learning "jumps".

In a way, the *J(θ)* function can be understood as an equivalent to the ***loss function*** of classic deep supervised learning.

The main obstacle is that *G_t* relies on the results of the future timesteps and we cannot analitically calculate *J(θ)*, much less its derivative. Therefore, many different methods have been developed to approximate it. We have chosen to implement 3 of these methods, described below: *REINFORCE*, *REINFORCE with Baseline* and *Proximal Policy Optimization*.

### REINFORCE (Vanilla Policy Gradient)

The first algorithm we implemented is also the simplest policy-based RL algorithm: REINFORCE, a.k.a. Vanilla Policy Gradient.

REINFORCE is a mathematical version or *trial-and-error*: we essentially attempt to make the car complete a lap around the track one, then train the network on that lap attempt, and try again for as many times as we seem necessary.

On each lap attempt, the agent reads the state from the environment and calls a `select_action` function that samples an action from a set of action probabilities calculated by the policy network; this sampled action is then fed to the environment in order to output a reward and the next state. Each reward output by the environment is stored in a buffer, as well as all the action probability (in logarithmic form) for the action that resulted in that reward. When the lap attempt finishes and all rewards are collected,  we proceed update the policy network's weights by following this formula:

![Reinforce formula](/readme_media/reinforce.jpg)

In REINFORCE, the approximation of the gradient of the objective function is simply the product of Gt (the sum of all future rewards from that timestep *t*) and the logarithm of the action probability. This can be interpreted as the formula of the Policy Gradient Theorem and using Monte Carlo (samples) to obtain approximate values.

By storing the reward and the action probability of each step of the lap attempt in a buffer, we can easily calculate *G_t* for any given timestep and multiply it with the logarithm of the action probability that we obtained when selecting the action. We calculate this value for all timesteps in a buffer and then apply backpropagation on it to update the policy parameters. We also update the running reward right before updating the policy.

The REINFORCE algorithm is easy to implement but has 2 big shortcomings:
* Very high variance: REINFORCE is capable of finding very good policy parameters but convergence is very slow because each lap attempt can vary greatly. Finding a good running reward without many thousands of lap attempts requires lots of luck in order to get lap attempts with good rewards.
* Slow learning: REINFORCE only updates the policy network once, at the end of each lap attempt. This requires us to run as many lap attempts as policy updates we want to have, thus slowing down the rate of learning even further.

### REINFORCE with Baseline

REINFORCE with Baseline adds a *baseline* variable to the objective formula in order to attempt to reduce the high variance issue. A good baseline variable is the *state value function V(s_t)*, which is defined as the expected returns (*expected G_t*) starting at the state *s_t* following a policy *π_θ*. The updated formula for updating the policy parameters is as follows:

![RL with Baseline formula](/readme_media/baseline_formula.jpg)

In order to calculate *V(s_t)* we need to make a small change to our policy network: instead of having a single output for the action, we will have 2 separate fully connected layers at the end; the `actor_head` calculates the action probabilities and the `critic_head` calculates the state value. The `select_action` method samples an action just like in regular REINFORCE, and all the rewards, actions and state values are stored in a buffer for each step in the lap attempt.

After completing the lap attempt and updating the running reward, we proceed to update the policy parameters following the formula described above by using our stored values in the buffer, similarly to the REINFORCE implementation.

REINFORCE with Baseline is an easy method to implement because it only requires small modifications to REINFORCE and it also shows good results with fewer lap attempts.

### Proximal Policy Optimization (PPO)

The final method we implemented is Proximal Policy Optimization, a much more complex algorithm than the previous 2 while at the same time being a simplification of other even more complex methods.

PPO tries to solve the 2 shortcomings of REINFORCE by allowing us to update the policy network multiple times in a single lap attempt as well as controlling the updates by comparing the results between the original policy and the updated policy with the purpose of both improving training rate and avoiding high variance and noise.

The formula for updating the policy parameters is much more elaborate:

![PPO formula](/readme_media/ppoformula.jpg)

There are 3 important components to this formula:

* The *clip loss L^CLIP* controls the policy update by limiting the update rate: if a good action is likelier under the new policy or if a bad action is likelier under the old policy, we limit the amount of credit that the new policy gets, thus making it more "pessimistic" and forcing the model to take more cautionary learning steps. It could also be understood as the *actor loss*, because it deals with the loss of the `actor_head` of our policy network.
* The *value function loss L^VF* could also be understood as the *critic loss* because it calculates the loss of the state value that the `critic_head` outputs. It's a simple mean squared error formula that we combine with the clip loss with a discount factor hyperparameter to bring both losses to the same order of magnitude.
* Finally, the *entropy term S\[π_θ\](s_t)* is added in order to encourage exploring different policies and regulated with an additional discount factor hyperparameter.

We start the algorithm by filling a *transition memory* that stores tuples that represent each step (or transition) when attempting a lap: state, chosen action probability, reward, state value and the next state returned by the environment. The memory is able to store more transitions that happen in a regular lap attempt, so we do as many lap attempts as necessary in order to fill the memory.

Once the memory is full, we compute a few additional values needed for the PPO formula and then proceed to the actual training steps:
1. For K epochs do:
   1. For random mini_batch in transition memory:
      1. Compute PPO loss using the formula above.
      2. Update the policy weights
2. Discard transition memory.

And repeat for as many episodes as needed.

PPO took by far the longest time of all 3 methods to implement and we stalled many times due to debugging issues, with unsatisfactory results. We have been testing our implementation as late as the weekend before the day of our defense presentation.

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

## Pending stuff

## References and notes

* [Juanjo Nieto's *Policy Gradient Methods* slides](https://docs.google.com/presentation/d/1LKe3pLUIphKDytIuYF8LnM23trjHoWoAw22rDT4kCrs/edit)
* [Víctor Campos' *Policy Gradients & Actor-Critic methods* slides](https://docs.google.com/presentation/d/1LBcfpJsOZlb5337-x2nqwqm0boay00HQELHlA-moUs8/edit)
* [OpenAI's *Proximal Policy Optimization Algorithms* paper](https://arxiv.org/pdf/1707.06347.pdf)
