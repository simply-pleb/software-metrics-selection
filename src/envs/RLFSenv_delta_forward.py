from src.errors import sammon_error
import numpy as np


class RLFSEnvDeltaForward:
    def __init__(
        self,
        state_size,
        data,
        max_features=None,
        error_f=sammon_error,
    ):
        self.state_size = state_size  # Length of the state vector
        self.state = np.zeros(
            self.state_size, dtype=bool
        )  # Initialize state as all False
        self.error_f = error_f
        self.data = data
        self._state_prev_error = error_f(data, self.state)
        # self.init_error = error_f(data, self.state)
        self.cur_num_features = 0
        self.max_features = max_features if max_features is not None else state_size

    def get_done(self):
        return self.cur_num_features >= self.max_features

    def get_reward(self, changed_state):
        # if not changed_state:
        #     return -1
        cur_error = self.error_f(self.data, self.state)
        delta = self._state_prev_error - cur_error # maximize delta (delta > 0)
        self._state_prev_error = cur_error
        return delta

    def reset(self):
        """Reset the environment state at the start of each episode."""
        self.state = np.zeros(self.state_size, dtype=bool)
        self._state_prev_error = self.error_f(self.data, self.state)
        self.cur_num_features = 0
        return self.state

    def step(self, action):
        """
        Perform the chosen action in the environment.

        Args:
            action (int): The index in the state to be set to True.

        Returns:
            state (np.array): Updated state after the action.
            reward (float): Reward for taking the action.
            done (bool): Whether the episode has ended.
            info (dict): Additional info, if any (empty here).
        """
        # Set the state at the action's index to True
        prev_state_at = self.state[action]
        self.state[action] = True

        changed_state = prev_state_at != self.state[action]
        self.cur_num_features = int(np.sum(self.state))

        # Reward logic
        reward = self.get_reward(changed_state)

        # Determine if the episode is done (e.g., all states set to True)
        done = self.get_done()
        # if done:
        #     print(done)

        return self.state, reward, done, {}
