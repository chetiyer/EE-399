# HW 7 : Snake Game analysis
Author: Chetana Iyer

This notebook looks at basic analysis of the snake Game following HW7 in EE 399A, Introduction to Machine Learning for Science and Engineering, a UW course designed by J. Nathan Kutz for Spring 2023.

## Abstract: 
In this assignment, we looked at our  your findings of game play as a function of exploration and the points assigned to death and eating the apple

## 1. Introduction
In this homework assignment, we explore the performance of the SHRED (SHallow REcurrent Decoder) neural network model for sea-surface temperature analysis. Firstly, we download the example code from the provided GitHub repository, which includes both the code and data. Secondly, we train the SHRED model using an LSTM/decoder architecture and visualize the results to assess its effectiveness. Next, we conduct a comprehensive analysis to evaluate the model's performance in relation to the time lag variable, noise levels (by adding Gaussian noise to the data), and the number of sensors employed. Through these analyses, we gain insights into the capabilities and limitations of the SHRED model for various scenarios in sea-surface temperature analysis.

## 2. Theoretical Background

The game is based off the concepts of **Reinforcement Learning** and the **MonteCarlo algorithm** 

### *Reinforcement learning* 
is a branch of machine learning where an artificial agent learns to make decisions by trial and error in an interactive environment. It takes inspiration from how humans and animals learn from rewards and punishments.
In this approach, the agent interacts with the environment and receives feedback in the form of rewards or penalties based on its actions. The goal is for the agent to learn the best strategy, called a policy, to maximize the cumulative reward it receives over time.
The agent learns by exploring different actions and observing the outcomes, gradually building its knowledge base. It improves its policy through trial and error, trying out different actions and updating its strategy based on the feedback from the environment.
A key idea in reinforcement learning is the balance between exploration and exploitation. Initially, the agent explores different actions to gather information and discover the most effective strategy. As it learns, it starts exploiting its existing knowledge by choosing actions that have previously resulted in high rewards. Finding the right balance between exploration and exploitation is crucial for efficient learning and adaptation.
Reinforcement learning finds applications in diverse areas like robotics, game playing, autonomous vehicles, and recommendation systems. By learning from experience and adapting to changing circumstances, reinforcement learning empowers artificial agents to make intelligent decisions and optimize their performance in complex environments.

### *The Monte Carlo algorithm* 
uses the concept of sampling and averaging returns to estimate the value of states or actions. The algorithm relies on the Law of Large Numbers and the concept of the Monte Carlo method.
The Monte Carlo algorithm starts by running multiple episodes or trajectories in the environment. During each episode, the agent takes actions and receives rewards until reaching a terminal state. The goal is to estimate the value of a particular state or action based on these sampled episodes.
To estimate the value, the Monte Carlo algorithm calculates the average return obtained from all the episodes that started from or took that state or action. The return is the cumulative sum of rewards received from the current time step until the end of the episode.
The value estimate for a state or action is updated incrementally as more episodes are sampled. The algorithm keeps track of the total return obtained for that state or action and the number of times it has been visited. The value estimate is then calculated as the average return:
Value_estimate = Total_return / Number_of_visits
As more episodes are simulated and the number of visits increases, the value estimate converges to the true value of the state or action. This convergence is guaranteed by the Law of Large Numbers, which states that the average of a large number of samples will approach the expected value.
The Monte Carlo algorithm iteratively updates the value estimates after each episode and continues this process until the estimates stabilize or converge. Once the value estimates have converged, the agent can use these estimates to make informed decisions on how to act in the environment.
Overall, the Monte Carlo algorithm utilizes the idea of sampling and averaging returns to estimate the value of states or actions. By running multiple episodes and updating the value estimates based on the collected returns, the algorithm converges to accurate estimations over time.

It consists of the following parameters: 
### **1** 
Number of episodes - which is the number of episodes or trajectories the algorithm will simulate to collect data. A higher number of episodes generally leads to more accurate value estimates but is also more computationally expensive. 
### **2** 
Epsilon - this is essentially a metric of exploration vs. exploitation. The Monte Carlo algorithm needs to strike a balance between exploration and exploitation. Exploration refers to trying out different actions to gather information, while exploitation involves leveraging existing knowledge to choose actions that have resulted in high rewards.
In the context of the snake game - this could be thought of prioritizing trying different paths vs. trying the path that is known to be safe 
### **3** 
Gamma - the discount factor, this value represents how much the snake cares about the immediate reward vs. future benefit 
It influences how the agent values delayed rewards and impacts the estimated returns. A discount factor closer to 1 considers future rewards more, while a value closer to 0 places more emphasis on immediate rewards.

### **4** 
Rewards represent the numerical feedback received by the agent from the environment based on its actions. The nature of rewards depends on the specific problem or environment being addressed. For example, in a game, rewards can be positive for winning or scoring points, negative for losing or making mistakes, or zero for neutral outcomes.
In this context, the reward values are for making losing moves, inefficient moves, efficient moves, and finally winning moves 

This assignment looks at the exploration of these different parameters and how 
it affects the snake's performance in the snake game


## 4. Analysis,Results & Conclusion

**Validation Error during training** ![here](https://github.com/chetiyer/EE-399/blob/main/download-2.png)

**Performance as a function Time Lag** ![here](https://github.com/chetiyer/EE-399/blob/main/download-1.png)

**Peformance as a function of Noise level** ![here](https://github.com/chetiyer/EE-399/blob/main/download.png)  

**Peformance as a function of Number of Sensors** ![here](https://github.com/chetiyer/EE-399/blob/main/download-3.png)  
