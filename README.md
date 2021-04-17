# Car Racing with RL - UPC AIDL winter 2021

This is the repo for the final project of the group under the supervision of Juanjo Nieto for the UPC School's Artificial Intelligence with Deep Learning postgraduate course, online edition of winter 2020-2021. The team members are:
* Juanjo Nieto (Advisor)
* Xavier Canal
* Rubén Martínez
* Alvaro Navas
* Jaime Pedret

The original goal of the project was to train a self-driving model that would allow us to have a vehicle of some sort drive itself through a circuit. One of our members, Rubén Martínez, owns a robot which is well suited for this task, but since the other members did not have easy access to it, it was decided that we would start by training a model in a OpenAI Gym environment which could be later be adapted to the robot, and the tasks were divided so that Rubén would work on the robot side and the rest would work on the Gym side.

In the end the original goal was too ambitious and the project ended up divided in 2 separate parts: the Gym part and the Robot part. This repo contains the Gym part; you may check out the robot part by visiting Rubén's repo at https://github.com/eldarsilver/DQN_Pytorch_ROS .

# Running the code **(NEEDS TO BE REDONE)**

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

# Reinforcement Learning and Car Racing

Reinforcement Learning (RL) is a *computational approach to goal-directed learning form interaction that does not rely on expert supervision* [(quote)](https://mitpress.mit.edu/books/reinforcement-learning-second-edition). In other words, it's the branch of Machine Learning that tries to achieve a task by using an active agent that reads data from the environment and a "teacher" that gives an extrinsic reward to the model in order to teach it when it's doing well. The agent gets a ***state*** from the environment and performs an ***action*** on the environment, which is then either rewarded, punished or ignored; then the agent gets a new state and the cycle repeats.

![RL basic diagram](/readme_media/RL.jpg)

OpenAI Gym is a popular framework for training Reinforcement Learning models. Gym provides a set of pre-made environments that allows students and researchers to test different approaches to solve the tasks proposed by each particular environment. One of these environments is [Car Racing](https://gym.openai.com/envs/CarRacing-v0/), which provides an 8-bit-videogame-like environment in which a car appears on a randomly generated circuit. The task to be achieved in this environment is to teach the car to drive itself in order to finish a lap.

![Agent choosing random actions](/readme_media/randomagent.gif)

The Car Racing environment outputs a ***state*** consisting on a 96x96 RGB image that displays the car and the track from a top-down view, as well as an additional black bar at the bottom of the image which contains various car sensor info. The environment expects an ***action*** input which consists of an array with 3 floating point numbers which represent turning direction (from -1 to 1, representing left to right), throttle (from 0 to 1, representing no throttle to full throttle) and brake (from 0 to 1 too, representing no brake to full brakes) inputs for the car to take. After receiving the action, the environment will return a ***reward*** as well as a new ***state*** consisting of an updated image that reflects the updated position of the car in the track. The environment will also output a *done* boolean value that will be `True` when the car finishes a lap or when it drives outside of the boundaries of the environment.

The default reward is a floating point value that may be positive or negative depending on the performance of the car on the track. The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles visited in the track. For example, if you have finished the track in 732 frames, the reward will be 1000 - 0.1 * 732 = 926.8 points. The task is considered finished when the agent consistently gets more than 900 points, but the definition of "consistently" is undefined by the environment and it's left for the developer to define it when implementing a solution.

The Car Racing environment features a physics model that affects the behaviour of the car. The car has rear-wheel propulsion and has very high acceleration, which makes it very easy for the car to oversteer and drift, making it very hard for a human player to regain control once the car starts skidding.

# First steps

We decided to initially approach the task by using policy-based RL methods, starting from REINFORCE-Vanilla Policy Gradient and implementing more sophisticated algorithms as we understood the behavior, advantages and shortcomings of each algorithm. Our chosen library was PyTorch due to our familiriaty with it and its ease of use.

Before implementing any algorithm, however, we knew from our classes and from additional research that using the vanilla environment as-is would be inefficient. The OpenAI Gym framework allows the use of *wrappers*, pieces of code that "wrap" the default environment in order to alter the outputs to make them more convenient for our purposes. Gym already provides some wrappers but we also implemented a few. The wrappers we ended up using were:
* Monitor: one of Gym's provided wrappers. It provides an easy way to record the output of the environment to a video file.
* GrayScaleObservation: another provided wrapper; it transforms RGB images to monochrome. Useful for reducing dimensionality in cases where the additional RGB info does not provide useful info compared to black and white images.
* FrameStack: another provided wrapper; FrameStack allows us to "stack" frames (states) in order to create a "mini-batch" of sorts for more efficient training.
* FrameSkipper: an original wrapper; FrameSkipper allows us to "skip" frames: instead of choosing an action for each frame, we use the same action for all of the skipped frames. This allows us to reduce the amount of actions we need to calculate.
* EarlyStop: an original wrapper; when used, the environment will output `done = True` in additional circunstances besides the default ones, such as getting a negative average reward. This allows us to stop the execution early and train with more episodes.

***NEEDS REVIEW*** There is also the issue of defining what "consistently" means when trying to get a reward of "consistently more than 900 points". We settled on calculating a *running reward* that accumulates the previously obtained rewards and calculates an average of sorts that represents the reward you can expect from the model at a specific stage of training, which is calculated as `running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward`, where `episode_reward` is the reward we obtain for each lap attempt.

For all of our experiments, the chosen optimizer was Adam, since it seems to be the default optimizer for pretty much any task. We did not experiment with additional optimizers.

In order to simplify the code, we decided to have a discrete set of actions initially. The Car Racing environment accepts floating point values as input, which makes it possible to finetune the car driving experience by having continuous action values, but discrete separate actions allows us to understand the environment better by letting us experiment with the action values and simplifies the code by not having to calculate the additional probability distributions needed to calculate the continuous values. Our goal was to code continuous values as a later feature in order to compare experiments but unfortunately we did not have the time to do so.

# Deep neural network

The next step is defining a deep neural network architecture that can process the state.

As previously said, the Car Racing environment outputs a state that consists of a 96x96 pixel RGB image. The obvious choice then is to use some sort of model based on convolutional layers to extract the image features and work with them.

The state is a very small image composed with very simple graphics with flat colors, so a complex network with dozens of layers isn't needed. We designed a very simple model with 3 convolutional layers intercalated with 3 pooling layers and 3 fully connected layers at the end.

![Our model](/readme_media/policy.png)

We experimented with different network configurations and we ended up with slight differences for each implemented RL method, but the basic structure shared among them is the one described above.

# Policy-based RL

(This section explains a few theory concepts in order to give anyone who visits the repo additional context and help her understand our code; most details about these algorithms such as the Policy Gradient Theorem or detailed explanations of the formulas and how to obtain them won't be covered for brevity's sake as well as for being outside the scope of this README file. We encourage anyone who wishes to learn more about Reinforcement Learning to check the linked resources listed at the end of the file).

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

## REINFORCE (Vanilla Policy Gradient)

The first algorithm we implemented is also the simplest policy-based RL algorithm: REINFORCE, a.k.a. Vanilla Policy Gradient.

REINFORCE is a mathematical version or *trial-and-error*: we essentially attempt to make the car complete a lap around the track one, then train the network on that lap attempt, and try again for as many times as we seem necessary.

On each lap attempt, the agent reads the state from the environment and calls a `select_action` function that samples an action from a set of action probabilities calculated by the policy network; this sampled action is then fed to the environment in order to output a reward and the next state. Each reward output by the environment is stored in a buffer, as well as all the action probability (in logarithmic form) for the action that resulted in that reward. When the lap attempt finishes and all rewards are collected,  we proceed update the policy network's weights by following this formula:

![Reinforce formula](/readme_media/reinforce.jpg)

In REINFORCE, the approximation of the gradient of the objective function is simply the product of Gt (the sum of all future rewards from that timestep *t*) and the logarithm of the action probability. This can be interpreted as the formula of the Policy Gradient Theorem and using Monte Carlo (samples) to obtain approximate values.

By storing the reward and the action probability of each step of the lap attempt in a buffer, we can easily calculate *G_t* for any given timestep and multiply it with the logarithm of the action probability that we obtained when selecting the action. We calculate this value for all timesteps in a buffer and then apply backpropagation on it to update the policy parameters. We also update the running reward right before updating the policy.

The REINFORCE algorithm is easy to implement but has 2 big shortcomings:
* Very high variance: REINFORCE is capable of finding very good policy parameters but convergence is very slow because each lap attempt can vary greatly. Finding a good running reward without many thousands of lap attempts requires lots of luck in order to get lap attempts with good rewards.
* Slow learning: REINFORCE only updates the policy network once, at the end of each lap attempt. This requires us to run as many lap attempts as policy updates we want to have, thus slowing down the rate of learning even further.

## REINFORCE with Baseline

REINFORCE with Baseline adds a *baseline* variable to the objective formula in order to mitigate the high variance issue. A good baseline variable is the *state value function V(s_t)*, which is defined as the expected returns (*expected G_t*) starting at the state *s_t* following a policy *π_θ*. The updated formula for updating the policy parameters is as follows:

![RL with Baseline formula](/readme_media/baseline_formula.jpg)

In order to calculate *V(s_t)* we need to make a small change to our policy network: instead of having a single output, we will have 2 separate fully connected layers at the end; the `actor_head` calculates the action probabilities and the `critic_head` calculates the state value. The `select_action` method samples an action just like in regular REINFORCE, and all the rewards, actions and state values are stored in a buffer for each step in the lap attempt.

After completing the lap attempt and updating the running reward, we proceed to update the policy parameters following the formula described above by using our stored values in the buffer, similarly to the REINFORCE implementation.

REINFORCE with Baseline is an easy method to implement because it only requires small modifications to REINFORCE and it also shows good results with fewer lap attempts.

## Proximal Policy Optimization (PPO)

The final method we implemented is Proximal Policy Optimization, a much more complex algorithm than the previous 2 while at the same time being a simplification of other even more complex methods.

PPO tries to solve the 2 shortcomings of REINFORCE by allowing us to update the policy network multiple times in a single lap attempt as well as controlling the updates by comparing the results between the original and the updated policies with the purpose of both improving training rate and avoiding high variance and noise.

The formula for updating the policy parameters is much more elaborate:

![PPO formula](/readme_media/ppoformula.jpg)

There are 3 important components to this formula:

* The *clip loss L^CLIP* controls the policy update by limiting the update rate: if a good action is likelier under the new policy or if a bad action is likelier under the old policy, we limit the amount of credit that the new policy gets, thus making it more "pessimistic" and forcing the model to take more cautionary learning steps. It could also be understood as the *actor loss*, because it deals with the loss of the `actor_head` of our policy network.
* The *value function loss L^VF* could also be understood as the *critic loss* because it calculates the loss of the state value that the `critic_head` outputs. It's a simple mean squared error formula that we combine with the clip loss with a discount factor hyperparameter to bring both losses to the same order of magnitude.
* Finally, the *entropy term S\[π_θ\](s_t)* is added in order to encourage exploring different policies and regulated with an additional discount factor hyperparameter.

We start the algorithm by filling a *transition memory* that stores tuples that represent each step (or transition) when attempting a lap: state, chosen action probability, reward, entropy, state value and the next state returned by the environment. The memory is able to store more transitions that happen in a regular lap attempt, so we do as many lap attempts as necessary in order to fill the memory.

Once the memory is full, we compute a few additional values needed for the PPO formula and then proceed to the actual training steps:
1. For K epochs do:
   1. For random mini_batch in transition memory:
      1. Compute PPO loss using the formula above.
      2. Update the policy weights by backpropagating through the loss.
2. Discard transition memory.

And repeat for as many episodes as needed.

PPO took by far the longest time of all 3 methods to implement and we stalled many times due to debugging issues, with unsatisfactory results. We have been testing our implementation as late as the weekend before the day of our defense presentation.

# Development history and milestones

## Milestone #1: REINFORCE implementation

Our first milestone was reached when we managed to complete the REINFORCE implementation and start running experiments. Xavi was already familiar with GitHub and taught the rest of us how to work with git and create pull requests to incorporate features and we all tried to figure out how the REINFORCE algorithm works and how to implement it.

However, we stumbled with our implementation due to both theory misunderstandings and code bugs. Xavi also found a memory leak in one of OpenAI's Gym libraries which caused our experiments to run out of memory, so we were forced us to fork them and fix them in order to successfully run them.

We expected to have very slow training but we encountered no training at all. Early on we had a run which managed to get good results but we hadn't yet implemented random seed presetting and we could not replicate the experiment. We started adding variables to TensorBoard to study what was happening and we discovered that the network entropy was collapsing very quickly.

We decided to move on to implementing REINFORCE with Baseline as a way to deal with our frustration. We already knew that REINFORCE wasn't likely to solve the environment, so we decided to move on to the next milestone in order to further our progress and to get a better understanding of policy gradients.

## Milestone #2: REINFORCE with Baseline and first insights

Implementing REINFORCE with Baseline turned out to be one of the most productive tasks because it forced us to reevaluate many of our original design decisions:

* We discovered a fundamental mistake in the way we were dealing with action probabilities and calculating the fina loss. We managed to fix in [in this commit](https://github.com/xeviknal/aidl-2021-wo-rl/commit/907af7a8043a6b540111ea4c833a2cae0a69c23d). Surprisingly, the good results we had gotten initially in the early lucky REINFORCE run had been done with a LogSoftmax final activation function rather than a regular Softmax; we still do not understand how it managed to train.
* After some input from Juanjo, we realized that the set of discrete actions we had initially chosen was far from ideal, because we had not given it enough thought. We decided to create different action sets, each one with varying levels of granularization.
* Again, after some input from Juanjo, we discovered a small difference in the code he had managed to run and get good results: the learning rate. We had mindlessly chosen a learning rate of 0.1 instead of the more usual 0.01, which had an enormous impact on our results: we went from barely managing a running reward of 30 after 24k episodes to running rewards close to 700 in less than 5k episodes.
* We greatly improved our logging process. Sadly we lost the early results from the first milestone due to very big log files and GitHub issues and limitations; with the improvements we could now upload both network parameters and logs for each run and sotre them in separate branches for ease of comparison.

After the great improvements, we could finally start running some experiments both with REINFORCE with Baseline as well as with BASELINE.

1. [The first experiment in which we discovered the vast differences in training performance when reducing the learning rate](https://github.com/xeviknal/aidl-2021-wo-rl/pull/28)
2. The second experiment, in which we proceed to test our different action sets with our fixed policy network and learning rate. We wanted to test the effects of the different action sets: our first conclusions were that having a larger set of possible actions seemed to help our model reach higher rewards.
   1. [Action set #0](https://github.com/xeviknal/aidl-2021-wo-rl/pull/29) (largest action set).
   2. [Action set #1](https://github.com/xeviknal/aidl-2021-wo-rl/pull/30).
   3. [Action set #2](https://github.com/xeviknal/aidl-2021-wo-rl/pull/31).
   4. [Action set #3](https://github.com/xeviknal/aidl-2021-wo-rl/pull/32) (identical to action set #2 except for an additional "no action" action to allow the network to let the car move by inertia).
3. The third experiment, in which we tweaked our policy network after Juanjo's input: he suggested that there might be a possibility of further improvement by adding more fully connected layers to our `actor_head` and `critic_head`; we removed one layer from the main network body and moved it to the heads. We were excited by our preliminar results due to quick reward growth but the later experiments all suffered from entropy collapse.
   1. [Initial test](https://github.com/xeviknal/aidl-2021-wo-rl/pull/33), which managed our highest reward yet.
   2. [Action set #0](https://github.com/xeviknal/aidl-2021-wo-rl/pull/34).
   3. [Action set #1](https://github.com/xeviknal/aidl-2021-wo-rl/pull/35).
   4. [Action set #2](https://github.com/xeviknal/aidl-2021-wo-rl/pull/36).
4. The 4th experiment restored the missing fully connected layer to the policy network and tweaked the heads to have less parameters and have a smaller bottleneck. However, the results were disappointing due to entropy collapse.
   1. [Action set #0](https://github.com/xeviknal/aidl-2021-wo-rl/pull/39).
   2. [Action set #1](https://github.com/xeviknal/aidl-2021-wo-rl/pull/40).
   3. [Action set #2](https://github.com/xeviknal/aidl-2021-wo-rl/pull/41).
5. For our final experiment with REINFORCE with Baseline, we took inspiration from [other code we were studying for PPO](https://github.com/xtma/pytorch_car_caring) and completely redid our network: we dropped the pooling layers and added more convolutional layers. We also added a new action set with an extreme amount of granularization to test our theory regarding having better results by having more available actions. We experienced episodes of near entropy collapse with later recovery in all of them, so we ended up dropping this network architecture and decided to keep ours.
   1. [Action set #0](https://github.com/xeviknal/aidl-2021-wo-rl/pull/43) showed very promising results and managed to get peak running reward values of over 800, but after trying to improve the results by adding more training episodes, the reward became lower.
   2. [Action set #1](https://github.com/xeviknal/aidl-2021-wo-rl/pull/44) showed a very long quasi-entropy collapse but seemed to recover near the end.
   3. [Action set #2](https://github.com/xeviknal/aidl-2021-wo-rl/pull/46) had disappointing results.
   4. [Extra action set #4](https://github.com/xeviknal/aidl-2021-wo-rl/pull/47) had lower than expected rewards. The entropy never seemed to quasi-collapse as hard as in the other experiments but while it managed to get a peak running reward of nearly 450, it was much less than what the experiment with action set #0 showed.

Parallelly, we also started experimenting again with REINFORCE in order to understand why the entropy was collapsing so hard. Due to the success of the learning rate and the known inestabilty of the algorithm, [we decided to experiment with different learning rate values](https://github.com/xeviknal/aidl-2021-wo-rl/pull/37). The results were spectacular and with a very small learning rate of 1*10^(-5) we managed to get running rewards of more than 800.

After the varying results of the experiments with different actions, we also started analyzing what the actions were actually doing and how they impacted the results. By watching different results, we observed the following:
* The model never learns how to brake.
* Because each new visited tile increases the reward, the model tries to accelerate and go as fast as possible, which poses a problem for very tight turns because the model can't brake.
* When the car drifts out of the track and starts skidding on the grass, the model is able to bring the car back into the track as long as a track tile appears on screen. If the car drifts too far and the track disappears, the model cannot remember where the track as last seen and starts taking nonsensical actions.
* If the model manages to recover from a burnout and gets the car back on track, it has no way of telling the proper track orientation and it oftens starts going backwards through the track.

[We started experimenting with hybrid actions that combined both turning with acceleration and braking](https://github.com/xeviknal/aidl-2021-wo-rl/pull/45) in order to control the speed, and discovered that forcing a small amount of brakes while turning was an effective way of forcing the model to control its speed.

Here are the things we learnt at this stage:
* Hyperparameter tuning is crucial for Reinforcement Learning, as demonstrated by our learning rate experiments.
* Rewards affects greatly to the action probabilities. The model never learns how to brake because the motivation to do so is missing.
* We could "cheat" by designing an action set that limits the top speed of the car and induces actions that the model would otherwise never do, such as braking while turning.

***TODO add entropy explanation***

# Milestone #3: PPO

# Final experiments
## Method
### Hypothesis
### Experiment Setup
### Results
### Conclusions



## Pending stuff

There are several features and experiments that we wanted to implement but did not have the time for. In no particular order:
* Create a new wrapper that modifies the rewards by having it return a negative reward when going outside the track or when a drift is detected, with the hopes that this would add an incentive for the model to learn how to brake before turns.

## References and notes

* [Juanjo Nieto's *Policy Gradient Methods* slides](https://docs.google.com/presentation/d/1LKe3pLUIphKDytIuYF8LnM23trjHoWoAw22rDT4kCrs/edit)
* [Víctor Campos' *Policy Gradients & Actor-Critic methods* slides](https://docs.google.com/presentation/d/1LBcfpJsOZlb5337-x2nqwqm0boay00HQELHlA-moUs8/edit)
* [OpenAI's *Proximal Policy Optimization Algorithms* paper](https://arxiv.org/pdf/1707.06347.pdf)
* [Car Racing with PyTorch](https://github.com/xtma/pytorch_car_caring)
