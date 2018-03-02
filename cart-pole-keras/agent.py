import os
import random
from collections import deque

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


class Agent():
    def __init__(self, state_size: int, action_size: int) -> None:
        self.weight_backup = "cartpole_weight.h5"
        self.state_size = state_size
        self.action_size = action_size
        self.memory: deque = deque(maxlen=2000)
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.exploration_rate = 1.0
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.brain = self._build_model()

    def _build_model(self) -> Sequential:
        """Build the Neural Network Model

        Returns:
            Sequential -- the initialized model
        """

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        # input layer
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        # hidden layer
        model.add(Dense(24, activation='relu'))
        # output layer
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        # Load weight backup if they exist
        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model

    def save_model(self):
        """Save the model weghts on the disk
        """

        self.brain.save(self.weight_backup)

    def act(self, state: np.array) -> int:
        """Choose the next action based on the current state

        Arguments:
            state {np.array} -- the current state of the game

        Returns:
            int -- the next action in [0, 1] (left, right)
        """

        # Decide if we should pick a random action or a predicted one based
        # on the current exploration_rate (which is decaying with time)
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.brain.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state: np.array, action: int, reward: int, next_state: np.array, done: bool):
        """Add the current step to the memory

        Arguments:
            state {np.array} -- the initial state before taking action
            action {int} -- the action taken
            reward {int} -- the reward gained from taking the action
            next_state {np.array} -- the new state after the action was executed
            done {bool} -- True if the game is finished
        """

        # Append the current step to the memory. It will be used to learn from it
        # at the end of the epoch. See Agent.replay()
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size: int):
        """Train the model with data in memory. The learning in done only on the subset
        of the steps contained in memory to reduce load.

        Arguments:
            sample_batch_size {int} -- Number of steps choosen randomly to train the model
        """
        # Learn only if there are enough steps in memory
        if len(self.memory) < sample_batch_size:
            return
        # Select randomly `sample_batch_size` from memory
        sample_batch = random.sample(self.memory, sample_batch_size)

        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.brain.predict(next_state)[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            # Train the model
            self.brain.fit(state, target_f, epochs=1, verbose=0)
        # Reduce the `exploration_rate` as we learn
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
