+++
title = 'Reinforcement Learning -- Multi-Armed Bandits'
date = 2024-02-01T23:06:48-06:00
draft = false
math = true
+++


# A k-armed Bandit Problem

Consider the problem, you are faced repeatedly with a choice among k different options, or actions. After each choice you receive a numerical reward chosen from a stationary probability distribution that depends on the action you selected. You objective is to maximize the expected total reward over some time period.

In a k-armed bandit problem, each of the k actions has an expected r mean reward given that action is selected. Let us call this the value of that action. 

we denote the action selected on time step $t$ as $A_t$ , and the corresponding reward as $R_t$ . The value then of an arbitrary action $a$ , denoted $q_*(a)$ , is the expected reward given that $a$ is selected: 

$q_*(a)\doteq \mathbb {E}[R_t | A_t = a]$

We denote the estimated value of action $a$ at time step $t$ as $Q_t(a)$. So we would like $Q_t(a)$ to be close to $q_*(a)$ .

**The greedy actions:** If you maintain estimates of the action values, then at any time step there is at least one action whose estimated value is greatest.

When you select one of these greedy actions, you are exploiting your current knowledge of the values of the actions. 

If you select one of the nongreedy actions, you are exploring, because this enables you to improve your estimate of the nongreedy action’s value. Exploitation may produce the greater total reward in the long run.

In any specific case, whether it is better to explore or exploit depends in a complex way on the precise values of the estimates, uncertainties, and the number of remaining steps. There are many sophisticated methods for balancing exploration and exploitation for particular mathematical formulations for the k-armed bandit and related problems.

# Action-value Methods

What is action-value methods ?

Answer: we begin by looking more closely at methods for estimating the values of actions and for using the estimates to make action selection decisions.

Recall that the true value of an action is the mean reward when that action is selected. One natural way to estimate this is by averaging the rewards actually received:

$$
Q_t(a) \doteq \frac {\text{sum of rewards when a taken prior to t}} {\text{number of times a taken prior to t}} = \frac {\sum\_{i=1} ^{t-1}R_i \cdot \mathbb {1}\_{A_i=a} } {\sum_{i=1} ^{t-1} \mathbb {1}\_{A_i = a}}
$$

Where $1_{predicate}$ denotes the random variable that is 1 if predicate is true and 0 if it is not.

If the denominator is zero, then we instead define $Q_t(a)$ as some default value, such as 0. As the denominator goes to infinity, by the law of large numbers, $Q_t(a)$ converges to $q_*(a)$.

We call this the sample-average method of estimating action values because each estimate is an average of the sample of relevant rewards. 

The simplest action selection rule is to select one of the actions with the highest estimated value, that is, one of the greedy actions as defined in the previous section.

If there is more than one greedy action, then a selection is made among them in some arbitrary way, perhaps randomly. 

So, the greedy action selection method:

$$
A_t \doteq \argmax_aQ_t(a)
$$

where $argmax_a$ denotes the action $a$ for which the expression that follows is maximized.

Greedy action selection always exploits current knowledge to maximize immediate reward; it spends no time at all sampling apparently inferior actions to see if they might really be better.

Reinforcement learning requires a balance between exploration and exploitation.

# Incremental Implementation

Let $R_i$ denote the reward received after the $i$ th selection of this action, and let $Q_n$ denote the estimate of its action value after it has been selected $n- 1$ times:

$$
Q_n \doteq \frac {R_1 + R_2  + ...+ R_{n-1}} {n - 1}
$$

The implementation would be to maintain a record of all the rewards and then perform this computation whenever the estimated value was needed.

Given $Q_n$ and the $n$ th reward, $R_n$, the new average of all $n$ rewards can be computed by:

$$
\begin{aligned}Q_{n + 1} &= \frac {1} {n} \sum_{i = 1} ^{n} R_i \\\ &= \frac {1} {n} (R_n + \sum_{i = 1} ^{n - 1} R_i) \\\ &= \frac {1} {n} (R_n + (n - 1) \frac {1} {n-1} \sum_{i = 1} ^{n - 1} R_i) \\\ &= \frac {1} {n} (R_n + (n-1)Q_n) \\\ &= \frac {1} {n} (R_n + nQ_n - Q_n) \\\ &= Q_n + \frac {1} {n}[R_n - Q_n]\end{aligned}
$$

Which holds even for $n = 1$, obtaining $Q_2 = R_1$ for arbitrary $Q_1$ . 

In processing the $n$ th reward for action $a$, the method uses the step-size parameter $\frac {1} {n}$ .

So, the general form is:

$$
NewEstimate \leftarrow OldEstimate + StepSize [ Target - OldEsimate]
$$

The expression $[ Target - OldEstimate]$ is an $error$ in the estimate. It is reduced by taking a step toward the Target. The target is presumed to indicate a desirable direction in which to move, though it may be noisy.

## A simple bandit algorithm

Initialize, for $a = 1$ to $k$:

$Q(a) \leftarrow 0$

$N(a) \leftarrow 0$

Loop forever: 

$$
\begin{aligned}
A &\leftarrow \begin{cases} argmax\_{a} Q(a) & \text{with probability 1 - }\varepsilon   
  \\\ \text{a random action } & \text{with probability } \varepsilon \end{cases} \\\ R &\leftarrow bandit(A) \\\ N(A) &\leftarrow N(A) + 1 \\\ Q(A) &\leftarrow Q(A) +  \frac {1} {N(A)} [R - Q(A)] \end{aligned}
$$

# Tracking a Nonstationary Problem

The averaging methods discussed so far are appropriate for stationary bandit problems, that is, for bandit problems in which the reward probabilities do not change over time. In such cases it makes sense to give more weight to recent rewards than to long-past rewards.

So, We are updating $Q_{n+1}$:

$$
Q_{n + 1} \doteq Q_n + \alpha[R_n - Q_n]
$$

Where the step-size parameter $\alpha \in (0, 1]$ is constant. This result in $Q_{n + 1}$  being a weighted average of past rewards and the initial estimate $Q_1$: 

$$
\begin{aligned} Q_{n + 1} &= Q_n + \alpha [R_n - Q_n] \\\ &= \alpha R_n + (1 - \alpha)Q_n \\\ &= \alpha R_n + (1 - \alpha)[\alpha R_{n - 1} + (1 - \alpha)Q_{n- 1}] \\\ &= \alpha R_n + (1 - \alpha) \alpha R_{n - 1} + (1 - \alpha)^2Q_{n - 1} \\\ &= \alpha R_n + (1 - \alpha) \alpha R_{n - 1} + (1 - \alpha)^2 \alpha R_{n - 2} + ... + (1 - \alpha)^{n - 1} \alpha R_1 + (1 - \alpha)^n Q_1 \\\ &= (1- \alpha)^n Q_1 + \sum_{i = 1}^n\alpha (1 - \alpha)^{n - i} R_i\end{aligned}
$$

We call this a weighted average because the sum of the weights is $(1 - \alpha)^n + \sum_{i = 1} ^n \alpha (1 - \alpha) ^{n - i} = 1$, as you can check for yourself.

Note that the weight, $\alpha(1 - \alpha) ^{n - i}$, given to the reward $R_i$ depends on how many rewards ago, $n - i$, it was observed. The quantity  $1 - \alpha$ is less than 1, and thus the weight given to $R_i$ decreases as the number of intervening rewards increases.

In fact, the weight decays exponentially according to the exponent on  $1 - \alpha$. (If $1 - \alpha = 0$, then all the weight goes on the very last reward, $R_n$, because of the convention that $0^0 = 1$ ). This is sometimes called an $\text{exponential recency-weighted average}$.

Let $\alpha_n(a)$ denote the step-size parameter used to process the reward received after the $n$ th selection of action $a$. The choice $\alpha_n(a) = \frac {1} {n}$ results in the sample-average method, which is guaranteed to converge to the true action values by the law of large numbers. But of course convergence is not guaranteed for all choices of the sequence {${\alpha_n(a)}$}. 

A well-known result in stochastic approximation theory gives us the conditions required to assure convergence with probability 1:

$$
\sum_{n = 1} ^{\infty} \alpha_n(a) = \infty \text{    and  } \sum_{n = 1}^{\infty} \alpha_n ^2(a) < \infty
$$

The first condition is required to guarantee that the steps are large enough to eventually overcome any initial conditions or random fluctuations.

The second condition guarantees that eventually the steps become small enough to assure convergence.

Note that both convergence conditions are met for the sample-average case, $\alpha(a) = \frac {1} {n}$, but not for the case of constant step-size parameter, $\alpha_n(a) = \alpha$ . In the latter case, the second condition is not met, indicating that the estimates never completely converge but continue to vary in response to the most recently received rewards.

# Optimistic Initial Values

We need to find the initial action-value estimates, $Q_1(a)$.

Initial action values can be used as a simple way to encourage exploration.

Initially, the optimistic method performs worse because it explores more, but eventually it performs better because its exploration decreases with time, this process called optimistic initial values. 

# Upper-Confidence-Bound Action Selection

Exploration is needed because there is always uncertainty about the accuracy of the action-value estimates. The greedy actions are those that look best at present, but some of the other actions may actually be better. $\varepsilon$-greedy action selection forces the non-greedy actions to be tried, but indiscriminately, with no preference for those that are nearly greedy or particularly uncertain. It would be better to select among the non-greedy actions according to their potential for actually being optimal, taking into account both how close their estimates are to being maximal and the uncertainties in those estimates.

One effective way of doing this is to select actions according to:

$$
A_t \doteq \argmax_a[Q_t(a) + c \sqrt{\frac {lnt} {N_t(a)}}]
$$

where $lnt$ denotes the natural logarithm of $t$, $N_t(a)$ denotes the number of times that action $a$ has been selected prior to time $t$, and the number $c > 0$ controls the degree of exploration. If $N_t$(a) = 0, then $a$ is considered to be a maximizing action.

The idea of this $\text{upper confidence bound}$ (UCB) action selection is that the square-root term is a measure of the uncertainty or variance in the estimate of $a$’s value. The quantity being max’ed over is thus a sort of upper bound on the possible true value of action $a$, with $c$ determining the confidence level. Each time $a$ is selected the uncertainty is presumably reduced:

$N_t$(a) increments, and, as it appears in the denominator, the uncertainty term decreases. 

On the other hand, each time an action other than $a$ is selected, $t$ increases but $N_t(a)$ does not; because $t$ appears in the numerator, the uncertainty estimate increases.

The use of the natural logarithm means that the increases get smaller over time, but are unbounded; all actions will eventually be selected, but actions with lower value estimates, or that have already been selected frequently, will be selected with decreasing frequency over time. 

# Gradient Bandit Algorithms

Although we have considered methods that estimate action values and use those estimates to select actions, it is not the only one possible. Maybe we can consider other methods that can learning a numerical preference for each action $a$, which we denote $H_t(a)$. The larger the preference, the more often that action is taken, but the preference has no interpretation in terms of reward. Only the relative preference of one action over another is important; If we add 1000 to all the action preferences there is no effect on the action probabilities, which are determined according to a soft-max distribution as follows:

$$
\begin {aligned} Pr{ \{A_t = a} \} \doteq \frac {e^{H_t(a)}} {\sum_{b=1} ^k e^{H_t(b)}} \doteq \pi_t(a) \end{aligned}
$$

We gave introduced a useful new notation, $\pi_t(a)$, for the probability of taking action $a$ at time $t$. Initially all action preferences are the same so that all actions have an equal probability of being selected.

There is a natural learning algorithm for this setting based on the idea of stochastic gradient ascent. On each step, after selecting action $A_t$ and receiving the reward $R_t$ , the action preferences are updated by:

$$
\begin{aligned} H_{t+1}(A_t) &\doteq H_t(A_t) + \alpha (R_t - \bar{R_t}) (1 - \pi_t(A_t)), and \\\ H_{t+1}(a) &\doteq H_t(a) - \alpha(R_t-\bar{R_t})\pi_t(a)), \text{for all } a \neq A_t,\end{aligned}
$$

Where $\alpha > 0$  is a step-siez paramter, and $\bar{R_t} \in \mathbb{R}$ is the average of all the rewards up through and including time $t$. The $\bar{R_t}$ term serves as a baseline with which the reward is compared.

If the reward is higher than the baseline, then the probability of taking $A_t$ int the future is increased, and id the reward is below baseline, then probability is decreased. The non-selected actions move in the opposite direction.

## The Bandit Gradient Algorithm as Stochastic Gradient Ascent

In exact gradient ascent, each action preference $H_t(a)$ would be incremented proportional to the increment’s effect on performance:

$$
H_{t+1}(a) \doteq H_t(a) + \alpha \frac {\partial {\mathbb E[R_t]}} {\partial H_t(a)}
$$

where the measure of performance here is the expected reward:

$$
\mathbb E[R_t] = \sum_x \pi_t(x)q_*(x)
$$

and the measure of the increment’s effect is the partial derivative of this performance measure with respect to the action preference. 

First, we take a closer look at the exact performance gradient:

$$
\begin{aligned} \frac {\partial {\mathbb E[R_t]}} {\partial {H_t(a)}} &= \frac {\partial} {\partial H_t(a)} [\sum_x \pi_t(x)q_\*(x)] \\\ &= \sum_x q_\*(x) \frac {\partial {\pi_t(x)}} {\partial {H_t(a)}} \\\ &= \sum_x (q_\*(x) - B_t) \frac {\partial {\pi_t(x)}} {\partial {H_t(a)}}
\end{aligned}
$$

where $B_t$, called the baseline, can be any scalar that does not depend on $x$. 

We can include a baseline here without changing the equality because the gradient sums to zero over all the actions, $\sum_x \frac {\partial {\pi_t(x)}} {\partial {H_t(a)}} = 0$ , as $H_t(a)$ is changed, some actions’ probabilities go up and some go down, but the sum of the changes must be zero because the sum of the probabilities is always one.

Next, we multiply each term of the sum by $\frac {\pi_t(x)} {\pi_t(x)}$:

$$
\frac {\partial {\mathbb E[R_t]}} {\partial H_t(a)} = \sum_x \pi_t(x)(q_*(x) - B_t) \frac {\partial {\pi_t(x)}} {\partial {H_t(a)}} \cdot \frac {1} {\pi_t(x)}
$$

The equation is now in the form of an expectation, summing over all possible values $x$ of the random variable $A_t$, then multiplying by the probability of taking those values. 

Thus:

$$
\begin{aligned} &= \mathbb E[(q_*(A_t) - B_t) \frac {\partial {\pi_t(A_t)}} {\partial {H_t(a)}} \cdot \frac {1} {\pi_t(A_t)}] \\\ &= \mathbb E[(R_t - \bar{R_t}) \frac {\partial {\pi_t(A_t)}} {\partial {H_t(a)}} \cdot \frac {1} {\pi_t(A_t)}],\end{aligned}
$$

where here we have chosen the baseline $B_t = \bar{R_t}$ and substituted $R_t$ for $q_*(A_t)$, which is permitted because $\mathbb {E[{R_t} | {A_t}]} = q_*(A_t)$. Shortly we will establish that $\frac {\partial {\pi_t(x)}} {\partial {H_t(a)}} = \pi_t(x) (\mathbb 1_{a=x} - \pi_t(a))$, where $\mathbb 1_{a=x}$ is defined to be $1$ if $a = x$, else $0$. 

Assuming that for now, we have,

$$
\begin{aligned} &= \mathbb E[(R_t - \bar{R_t})\pi_t(A_t)(\mathbb1_{a=A_t} - \pi_t(a)) \cdot \frac {1} {\pi_t(A_t)}] \\\ &= \mathbb E[(R_t - \bar{R_t})(\mathbb 1_{a=A_t} - \pi_t(a))]. \end{aligned}
$$

Recall that our plan has been to write the performance gradient as an expectation of something that we can sample on each step, as we have just done, and then update on each step proportional to the sample. 

Substituting a sample of the expectation above for the performance gradient:

$$
H_{t+1}(a) = H_t(a) + \alpha(R_t - \bar{R_t}) (\mathbb 1\_{a=A_t} - \pi_t(a)), \text{ for all }a,  
$$

Since,

$$
H_{t+1}(a) \doteq H_t(a) - \alpha (R_t - \bar{R_t})\pi_t(a), \text { for all } a \neq A_t
$$

Thus it remains only to show that $\frac {\partial {\pi_t(x)}} {\partial {H_t(a)}} = \pi_t(x)(\mathbb1_{a=x} - \pi_t(a))$, as we assumed.

Recall the standard quotient rule for derivatives:

$$
\frac {\partial} {\partial x}[\frac {f(x)} {g(x)}] = \frac {{\frac {\partial f(x)} {\partial x}g(x) - f(x) \frac {\partial g(x)} {\partial x} } } {g(x)^2}.
$$

Using this, we can write

$$
\begin{aligned} \frac {\partial \pi_t(x)} {\partial H_t(a)} &= \frac {\partial} {\partial H_t(a)} \pi_t(x) \\\ &= \frac {\partial} {\partial H_t(a)} [\frac {e^{H_t(x)}} {\sum_{y=1} ^k e^{H_t(y)}}] \\\ &= \frac {\frac {\partial e^{H_t(x)}} {\partial H_t(a)}\sum_{y=1}^k e^{H_t(y)} - e^{H_t(x)} \frac {\partial \sum_{y=1} ^k e^{H_t(y)}} {\partial H_t(a)}} {(\sum_{y=1} ^k e^{H_t(y)})^2} \\\ &= \frac {\mathbb1_{a=x}e^{H_t(x)}} {\sum_{y=1} ^k e^{H_t(y)}} - \frac {e^{H_t(x)}e^{H_t(a)}} {(\sum_{y=1} ^k e^{H_t(y)})^2} \\\ &= \mathbb1_{a=x} \pi_t(x) - \pi_t(x)\pi_t(a) \\\ &= \pi_t(x)(\mathbb1_{a=x} - \pi_t(a)).\end{aligned}
$$

Tips: $\frac {\partial e^x} {\partial x} = e ^x$

We have just shown that the expected update of the gradient bandit algorithm is equal to the gradient of expected reward, and thus that the algorithm is an instance of stochastic gradient ascent. This assures us that the algorithm has robust convergence properties.

Note that we did not require any properties of the reward baseline other than that it does not depend on the selected action.