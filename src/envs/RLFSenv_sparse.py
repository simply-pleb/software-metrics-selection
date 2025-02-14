from src.errors import sammon_error
import numpy as np


class RLFSEnvSparse:
    def __init__(
        self,
        state_size,
        data,
        max_features=15,
        error_f=sammon_error,
    ):
        self.state_size = state_size  # Length of the state vector
        self.state = np.zeros(
            self.state_size, dtype=bool
        )  # Initialize state as all False
        self.error_f = error_f
        self.data = data
        self._state_prev_error = None  # error_f(data, self.state)
        self.init_error = error_f(data, self.state)
        self.cur_num_features = 0
        self.max_features = max_features

    def get_done(self):
        return self.cur_num_features >= self.max_features

    def get_reward(self):
        if self.get_done():
            # r = self.init_error - self.error_f(self.data, self.state)
            r = - np.log(self.error_f(self.data, self.state) + 1e-5)
            return r
        return 0.0

    def reset(self):
        """Reset the environment state at the start of each episode."""
        self.state = np.zeros(self.state_size, dtype=bool)
        self._state_prev_error = None  # error_f(data, self.state)
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
        # prev_state_at = self.state[action]
        self.state[action] = True

        # changed_state = prev_state_at != self.state[action]
        self.cur_num_features = int(np.sum(self.state))

        # Reward logic
        reward = self.get_reward()

        # Determine if the episode is done (e.g., all states set to True)
        done = self.get_done()

        return self.state, reward, done, {}
