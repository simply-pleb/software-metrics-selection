import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from src.envs import RLFSEnvDeltaBackward, RLFSEnvDeltaForward, RLFSEnvFull, RLFSEnvSparse
from src.rl import DQN, SARSA_N, PPO, REINFORCE
from src.errors import sammon_error

def get_data_frames():
    data_train = pd.read_csv("data/data_train.csv", sep=',')
    data_test = pd.read_csv("data/data_test.csv", sep=',')
    
    return data_train, data_test

def get_data_train_test(data_train, data_test):    
    X_train = data_train.drop(columns=["repository"], inplace=False)
    X_train = X_train.to_numpy()
    
    X_test = data_test.drop(columns=["repository"], inplace=False)
    X_test = X_test.to_numpy()
    
    return X_train, X_test

LOAD = "models/REINFORCE/"
LOAD = None

if __name__ == "__main__":
    data_train, data_test = get_data_frames()
    X_train, X_test = get_data_train_test(data_train, data_test)
    state_space = X_train.shape[1]
    action_space = X_train.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    env = RLFSEnvSparse(state_size=state_space, data=X_train, max_features=5)
    print(env.init_error)
    agent = REINFORCE.REINFORCEAgent(state_space, action_space, gamma=0.9, lr=0.005)
    
    if LOAD is None:
        episode_returns = agent.train(env=env, num_episodes=10000, max_steps=state_space)
        torch.save(agent.policy.state_dict(), "models/REINFORCE/policy_weights.pth")
        plt.plot(episode_returns)
        plt.show()
    else:
        agent.policy.load_state_dict(torch.load(LOAD + "policy_weights.pth"))

    INF_LOOP_CNT = 5
    env = RLFSEnvSparse(state_size=state_space, data=X_test, max_features=state_space)
    errors = []
    num_ftrs = []
    print("waiting...")
    for n in range(state_space):
        state = env.reset()
        # errors.append(sammon_error(X_test, state))
        state_cnt = 0 # int(np.sum(state))
        done = False
        print(f"n={n}")
        inf_loop_cnt = INF_LOOP_CNT
        while state_cnt < n and not done:
            if inf_loop_cnt > 0:
                action, action_prob = agent.select_action_deterministic(state)
            else:
                # print("+")
                action, action_prob = agent.select_action(state)
                # print(np.exp(action_prob.detach().numpy()))
                
            # print(action)
            next_state, _, done, _ = env.step(action)
            
            if int(np.sum(next_state)) > state_cnt:
                # print(state_cnt)
                inf_loop_cnt = INF_LOOP_CNT
                state_cnt = int(np.sum(next_state))
            else:
                inf_loop_cnt -= 1
            
            state = next_state
        print(data_test.drop(columns=['repository']).columns[state])
        error = sammon_error(X_test, state)
        errors.append(error)
        num_ftrs.append(n)
    print("done.")
        
    # plt.plot(errors)
    # plt.show()
    # Plot
    plt.plot(num_ftrs, errors, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Features')
    plt.ylabel('Sammon Error')
    plt.title('Sammon Error vs Number of Features')
    plt.grid(True)
    plt.xticks(np.arange(min(num_ftrs), max(num_ftrs) + 1, 5))  # Set tick frequency to every 5 units
    plt.show()

