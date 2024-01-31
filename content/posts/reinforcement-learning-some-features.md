+++
title = 'Reinforcement Learning Some Features'
date = 2024-01-31T14:25:34-06:00
draft = false
+++

# Reinforcement Learning Notes


Reinforcement learning takes the opposite tack, starting with a complete, interactive, goal-seeking agent. 

All reinforcement learning agents have explicit goals, can sense aspects of their environments, and can choose actions to influence their environments.

**Markov decision processes:**
	
	1.Sensation
	2.Action
	3.Goal

## Elements of Reinforcement Learning
### A reinforcement learning system: 
	1. a policy
	2. a reward signal
	3. a value function
	4. a model of the environment

#### A policy
A policy defines the learning agent’s way of behaving at a given time. A policy is a mapping from perceived states of the environment to actions to be taken when in those states. It corresponds to what in psychology would be called a set of stimulus-response rules or associations. The policy is the core of a reinforcement learning agent in the sense that it alone is sufficient to determine behavior. 

In general, policies may be stochastic, specifying probabilities for each action.

#### A reward signal
A reward signal defines the goal of a reinforcement learning problem. On each time step, the environment sends to the reinforcement learning agent at single number called the reward. The agent’s sole objective is to maximize the total reward it receives over the long run. The reward signal thus defines what are the good and bad events for the agent. The reward signal is the primary basis for altering the policy; if an action selected by the policy is followed by low reward, then the policy may be changed to select some other action in that situation in the future. 

In general, reward signals may be stochastic functions of the state of the environment and the actions taken.

#### A value function

The value of a state is the total amount of reward an agent can expect to accumulate over the future, starting from that states, values indicate the long-term desirability of states after taking into account the states that are likely to follow and the rewards available in those states. Whereas reward determine the immediate, intrinsic desirability of states after taking into account the states that are likely to follow and the rewards available in those states. To make a human analogy, rewards are somewhat like pleasure (if high) and pain (if low), whereas values correspond to a more refined and farsighted judgement of how pleased or displeased we are that our environment is in a particular state.

#### A model of the environment
It can mimics the behavior of the environment, or more generally, that allows inferences to be made about how the environment will behave.

Model-based methods: Methods for solving reinforcement learning problems that use models and planning.

Modern reinforcement learning spans the spectrum from low-level, trial-and-error learning to high-level, deliberative planning.

***Some key features of reinforcement learning methods:***

	1. There is the emphasis on learning while interacting with an environment.
	2. There is a clear goal, and correct behavior requires planning for foresight that takes into account delayed effects of one’s choices.

Reinforcement learning is a computational approach to understanding and automating goal-directed learning and decision making. It is distinguished from other computational approaches by its emphasis on learning by an agent from direct interaction with its environment, without requiring exemplary supervision or complete models of the environment.

Reinforcement learning uses the formal framework of Markov decision processes to define the interaction between a learning agent and its environment in terms of states, actions, and rewards. This framework is intended to be a simple way of representing essential features of the artificial intelligence problem. These features include a sense of cause and effect, a sense of uncertainty and nondeterminism, and the existence of explicit goals.

## References
1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction (2nd ed.).