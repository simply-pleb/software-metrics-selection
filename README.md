# Software Metrics Selection

Team:

- Ahmadsho Akdodshoev
- Jaffar Totanji
- Murhaf Alawir

## Abstract

This study explores unsupervised feature selection techniques,
Sequential Forward Selection (SFS) and the REINFORCE algorithm, to
optimize software metric subsets, addressing challenges like collinearity and noise in high dimensional datasets. Analyzing metrics from over 100 Python repositories, including cohesion, coupling, complexity, and main- tainability attributes, we used Sammon Error as the cost function to preserve data structure during dimensionality reduction. Results show both SFS and REINFORCE effectively minimized Sammon Error, identifying compact metric subsets that maintained predictive performance. For method metrics, Halstead Effort alone sufficed, while five metrics were optimal for class metrics. Our work highlights the value of feature selection in navigating high dimensional datasets and predicting software quality.

**paper in review**

## Methodology

### Data Collection

Dataset from:

- <https://github.com/angayunpa/EMProject>

### Optimization Criterion

The optimization criterion is minimization of Sammon Error, which is a metric for preserving the geometric structure of the data. Sammon Error is a distance-preserving transformation of the data, and is defined as:

$$E = \frac{1}{\sum_{i<j}{d_{ij}^*}}\sum_{i<j} \frac{\left(d_{ij}^* - d_{ij}\right)^2}{d_{ij}^*}$$

where $d_{ij}^*$ is the distance between the $i$-th and $j$-th data points in the original space, and $d_{ij}$ is the distance between the same points in the transformed space.

### Feature Selection - SFS

Sequential Forward Selection (SFS) is a greedy algorithm that iteratively adds the feature that most improves the performance of the model.

### Feature Selection - REINFORCE

REINFORCE is a policy gradient algorithm that uses the gradient of the policy to update the policy.

The problem of selecting $k$ feature was broken down into consecutively solving sub-problems $(1, 2, 2^2, ..., 2^{\lfloor\log_2 k\rfloor})$ to achieve stability for the policy $\pi(a \mid s, \theta)$ of the agent.

#### Environment

The state space is $\mathcal S = \{0,1\}^M$, where $M$ is the total number of features, $1$'s correspond to the chosen features and $0$'s correspond to the features left out. The action space is $\mathcal A = \{a\in\{0, 1\}^M:||a||_1=1\}$ where each action corresponds to selecting one feature. A state is changed every time a feature is selected for the first time.

The reward function was defined as:

$$
r = - \log(E + \varepsilon)
$$

where \(E\) is the Sammon Error and \(\epsilon\) is a small constant to avoid log(0).

### Failed Attempts

#### Environments

Environments with non-sparse reward functions were tried but did not perform better than environment with sparse reward function solved by REINFORCE. The following variants of environments were tried:

1. Reward equal to the negative of Sammon Error was given on every step.
2. Reward equal to the negative of change in Sammon Error was given on every step.
3. Reward equal to the negative of Sammon Error was given only when the episode ended.

negative reward is given on every step to encourage the agent to select features that minimize the Sammon Error due to the maximization of reward function.

#### Other RL Algorithms

PPO and DQN were also tried but they did not perform better than REINFORCE.

## Results

Results can be found in the corresponding Jupyter notebooks:

- [SFS](notebooks/test-sfs.ipynb)
- [REINFORCE](notebooks/test_reinforce.ipynb)
- [REINFORCE with Baseline](notebooks/test_reinforce_with_baseline.ipynb)
