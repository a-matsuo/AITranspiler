import math
import random
from typing import Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras

from rl.game import Game, encode_state


class MCTS:
    def __init__(self, qubits: int, network: keras.Model, config: Dict):
        """
        Initialize the Monte Carlo Tree Search (MCTS) instance.

        :param qubits: Number of qubits in the game.
        :param network: Neural network model for policy and value prediction.
        :param config: Configuration dictionary containing MCTS and game settings.
        """
        self.qubits: int = qubits
        self.network: keras.Model = network
        self.game = Game(qubits, config)

        mcts_settings = config["mcts_settings"]
        self.alpha: float = mcts_settings["dirichlet_alpha"]
        self.c_puct: float = mcts_settings["c_puct"]
        self.epsilon: float = mcts_settings["epsilon"]
        self.num_mcts_simulations: int = mcts_settings["num_mcts_simulations"]
        self.tau_threshold: int = mcts_settings.get("tau_threshold", 10)

        # Node statistics: keyed by string representation of states
        self.P: Dict[str, np.ndarray] = {}  # Policy distribution for each state
        self.N: Dict[str, np.ndarray] = {}  # Visit counts for each action
        self.W: Dict[str, np.ndarray] = {}  # Total value of each action
        self.next_states: Dict[str, list] = {}  # Next states for each action

        # Convert state to a hashable string key
        self.state_to_str = lambda state: "".join(map(str, state.astype(int).flatten()))

        # Parameters for temperature scheduling (if needed)
        self.initial_tau = 1.0
        self.final_tau = 0.1
        self.tau_decay_steps = 100
        self.current_episode = 0

    def update_temperature(self):
        """
        Update the temperature value for exploration based on the current episode.
        """
        decay_factor = min(1, self.current_episode / self.tau_decay_steps)
        return self.initial_tau * (1 - decay_factor) + self.final_tau * decay_factor

    def search(self, root_state, num_simulations: int, prev_action):
        """
        Perform MCTS searches starting from the root state.

        :param root_state: The initial state matrix.
        :param num_simulations: Number of MCTS simulations to run.
        :param prev_action: Previous action taken (if any).
        :return: A policy distribution over actions derived from visit counts.
        """
        s = self.state_to_str(root_state)
        tau = self.update_temperature()

        # If root not expanded, do so
        if s not in self.P:
            self._expand(root_state, prev_action)

        # Add Dirichlet noise to encourage exploration if needed
        valid_actions = self.game.get_valid_actions(root_state, prev_action)
        if self.alpha is not None and len(valid_actions) > 0:
            dirichlet_noise = np.random.dirichlet([self.alpha] * len(valid_actions))
            for a, noise in zip(valid_actions, dirichlet_noise):
                self.P[s][a] = (1 - self.epsilon) * self.P[s][a] + self.epsilon * noise
            self.P[s] = self.P[s] / np.sum(self.P[s])

        # Run MCTS simulations
        for _ in range(num_simulations):
            path = self._select(root_state, prev_action)
            leaf_state, leaf_prev_action = path[-1]

            # If leaf is terminal, get reward directly
            if self.game.is_done(leaf_state):
                value = self.game.get_reward(leaf_state, total_score=0)
            else:
                # Expand leaf if not expanded
                s_leaf = self.state_to_str(leaf_state)
                if s_leaf not in self.P:
                    value = self._expand(leaf_state, leaf_prev_action)
                else:
                    # If already expanded (should be rare), just use value from best action or 0 as fallback
                    # Typically we don't reach here if expansions are done once per leaf.
                    # But if we do, just treat as a terminal roll-in:
                    policy, val = self._predict(leaf_state)
                    value = val

            # Backpropagate value through the visited path
            self._backprop(path, value)

        # After simulations, build final policy distribution from visit counts
        visits = np.array([self.N[s][a] for a in range(self.game.action_space)])
        # Apply temperature
        if tau != 1.0:
            pi = visits ** (1 / tau)
        else:
            pi = visits
        pi = pi / np.sum(pi)

        return pi

    def _select(self, state, prev_action):
        """
        Selection phase: from the root, select actions until a leaf node is reached.
        Returns the path as a list of (state, prev_action) pairs.

        :param state: Current state from which selection starts (root).
        :param prev_action: Previous action from parent.
        :return: path: A list of (state, prev_action) visited during the descent.
        """
        path = []
        current_state = state
        current_prev_action = prev_action

        while True:
            s = self.state_to_str(current_state)
            if s not in self.P:
                # Leaf node found (not expanded)
                path.append((current_state, current_prev_action))
                return path

            # If terminal, return path immediately
            if self.game.is_done(current_state):
                path.append((current_state, current_prev_action))
                return path

            # Compute action selection using P, Q, UCB
            valid_actions = self.game.get_valid_actions(current_state, current_prev_action)
            if len(valid_actions) == 0:
                # No valid actions means terminal in some sense
                path.append((current_state, current_prev_action))
                return path

            N_s = np.sum(self.N[s])
            U = [
                self.c_puct * self.P[s][a] * math.sqrt(N_s) / (1 + self.N[s][a])
                for a in range(self.game.action_space)
            ]
            Q = [self.W[s][a] / self.N[s][a] if self.N[s][a] > 0 else 0
                 for a in range(self.game.action_space)]

            scores = [q + u if a in valid_actions else -np.inf
                      for a, q, u in zip(range(self.game.action_space), Q, U)]

            best_actions = np.where(scores == np.max(scores))[0]
            action = np.random.choice(best_actions)

            # Add to path
            path.append((current_state, current_prev_action))

            # Move to next state
            next_state = self.next_states[s][action]
            current_state = next_state
            current_prev_action = action

    def _expand(self, state, prev_action):
        """
        Expansion: For a leaf state, call the network to get its policy and value.
        Initialize N, W, and next_states. Return the state value for backprop.

        :param state: State to expand.
        :param prev_action: Previous action.
        :return: value predicted by the network.
        """
        s = self.state_to_str(state)
        policy, value = self._predict(state)

        self.P[s] = policy
        self.N[s] = np.zeros(self.game.action_space, dtype=np.float32)
        self.W[s] = np.zeros(self.game.action_space, dtype=np.float32)

        valid_actions = self.game.get_valid_actions(state, prev_action)
        next_s_list = []
        for a in range(self.game.action_space):
            if a in valid_actions:
                next_state, done, score = self.game.step(state, a, prev_action)
                next_s_list.append(next_state)
            else:
                next_s_list.append(None)
        self.next_states[s] = next_s_list

        return value

    def _predict(self, state):
        """
        Use the network to predict the policy and value for the given state.

        :param state: State matrix.
        :return: (policy, value) as numpy arrays.
        """
        input_state = encode_state(state, self.qubits)
        input_state = np.expand_dims(input_state, axis=0)
        policy_pred, value_pred = self.network.predict(input_state)
        policy = policy_pred[0]  # Assuming network outputs already in probability form or logits
        value = value_pred[0, 0] # Single scalar value
        # If policy needs softmax:
        # policy = tf.nn.softmax(policy_pred[0]).numpy()
        return policy, value

    def _backprop(self, path, value):
        """
        Backpropagate the value up the visited path.

        :param path: List of (state, prev_action) visited nodes.
                     The last node in path is the leaf for which we got 'value'.
        :param value: The evaluated value of the leaf state.
        """
        # path is a list of (state, prev_action). We need to go from leaf to root.
        # Each node in path except the leaf also has a chosen action that led to the next node.
        # The chosen action is stored in 'prev_action' of the *next* node.
        # Reconstruct the actions taken:
        for i in reversed(range(len(path))):
            state, prev_action = path[i]
            s = self.state_to_str(state)
            if i < len(path) - 1:
                # action taken at node i is the prev_action of node i+1
                _, next_prev_action = path[i + 1]
                action = next_prev_action
            else:
                # Leaf node doesn't lead to another node, so no action to update
                continue

            self.N[s][action] += 1.0
            self.W[s][action] += value
            # Q is implicitly updated as Q = W/N when used