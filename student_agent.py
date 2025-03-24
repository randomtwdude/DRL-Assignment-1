import numpy as np
import pickle
import random
import gym
import time

import torch
import torch.nn as nn
import torch.optim as optim

def softmax(x):
    exp_x = torch.exp(x - torch.max(x))
    return exp_x / exp_x.sum()

class Policy:
    def __init__(self, state_size, action_size, lrate=0.1):
        self.state_size = state_size
        self.action_size = action_size

        self.policy_table = torch.zeros(*state_size, action_size, requires_grad = True)
        self.optimizer = optim.Adam([self.policy_table], lr = lrate)
        self.criterion = nn.CrossEntropyLoss()

    def get_action(self, state):
        return np.random.choice(np.arange(self.action_size), 1, p = softmax(self.policy_table[state]).detach().numpy())

    def update(self, gamma, trajectory):
        # calculate rewards
        G = 0
        rewards = []
        for _, _, reward in reversed(trajectory):
            G = reward + gamma * G
            rewards.append(G)
        rewards.reverse()
        rewards = torch.tensor(rewards)

        policy_loss = 0
        for (state, action, _), r in zip(trajectory, rewards):
            p_values = self.policy_table[state].unsqueeze(0)
            action = torch.tensor(action)
            policy_loss += self.criterion(p_values, action) * r

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

class Memory:
    def __init__(self):
        self.stands = [0] * 4
        """
        0: unknown, 1: nothing, 2: alice, 3: dest
        """
        self.target = 0
        self.has_alice = False

    def get(self, obs):
        """
        (target_dir (3^2), obstacles (2^4), at alice, at dest., has alice)
        """
        # target direction
        dir_y, dir_x = obs[2 + self.target * 2] - obs[0], obs[3 + self.target * 2] - obs[1]
        """
        print("------")
        print(f"obs {obs}")
        print(f"target {self.target}, ({obs[0]}, {obs[1]}) -> ({obs[2 + self.target * 2]}, {obs[3 + self.target * 2]}), dir {dir_y}, {dir_x}")
        print("-----")
        """

        # stand
        on_stand = get_nearby_stands(obs)
        if on_stand is None:
            at_alice = at_dest = False
        else:
            at_alice = obs[14]
            at_dest  = obs[15]

        self._update_stands(on_stand, at_alice, at_dest)
        return (np.sign(dir_y) + 1, np.sign(dir_x) + 1, *obs[10:14], int(at_alice), int(at_dest), int(self.has_alice))

    def _update_stands(self, stand, alice, dest):
        if stand is None or self.stands[stand] != 0:
            return
        if dest:
            self.stands[stand] = 3
        elif alice:
            self.stands[stand] = 2
        else:
            self.stands[stand] = 1

    def update_target(self):
        try:
            if self.has_alice:
                self.target = self.stands.index(3)
            else:
                self.target = self.stands.index(2)
        except ValueError:
            self.target = self.stands.index(0)

def get_nearby_stands(obs):
    # we are on this stand exactly
    on_stand = None

    counter = 0
    taxi_pos = obs[0:2]
    for stand_pos in zip(*[iter(obs[2:10])] * 2):
        x_range = abs(stand_pos[1] - taxi_pos[1])
        y_range = abs(stand_pos[0] - taxi_pos[0])
        if x_range + y_range == 0:
            on_stand = counter
        counter += 1

    return on_stand

from simple_custom_taxi_env import SimpleTaxiEnv
def train_driver(lrate = 0.0005, gamma = 0.98, episode_total = 2000):
    print("Initilizing the strongest (table) model!")

    # set up training environment
    env = SimpleTaxiEnv(fuel_limit = 5000)

    state_size  = (3, 3, 2, 2, 2, 2, 2, 2, 2) # magic
    action_size = 6

    daiyousei = Policy(state_size, action_size) # dumb
    dys_grades = []

    start_time = time.time()

    for episode in range(episode_total):

        episode_reward = 0
        trajectory = []

        mem = Memory()
        obs, _ = env.reset()

        done = False
        unexplored_stands = 4
        while not done:
            state = mem.get(obs)
            action = daiyousei.get_action(state)
            obs, reward, done, _ = env.step(action)

            # STOP GOING BACK AND FORTH
            reward -= 1
            # pick up
            if not mem.has_alice and env.passenger_picked_up:
                reward += 60
                mem.has_alice = True
            # don't drop
            if mem.has_alice and not env.passenger_picked_up:
                reward -= 200
                mem.has_alice = False
            # well done
            if done and env.current_fuel > 0:
                reward += 1000
            # reward for exploring stands
            if mem.stands.count(0) < unexplored_stands:
                reward += 40
                unexplored_stands -= 1

            episode_reward += reward
            trajectory.append((state, action, reward))
            mem.update_target()

        daiyousei.update(gamma, trajectory)
        dys_grades.append(episode_reward)

        REPORT_INTERVAL = 50
        if (episode + 1) % REPORT_INTERVAL == 0:
            print(f"[{(time.time() - start_time):.0f}s] Episode {episode + 1}/{episode_total}: average grade {np.mean(dys_grades[-REPORT_INTERVAL:]):.0f}")
            start_time = time.time()

    np.save("daiyousei.npy", daiyousei.policy_table.detach().numpy())

"""
The real thing
"""
run_mem = Memory()
def get_action(obs):
    daiyousei = np.load("daiyousei.npy")

    # update memory
    state = run_mem.get(obs)

    # decide
    pick = np.argmax(daiyousei[state])

    # did we pick up
    if state[0] == 1 and state[1] == 1 and state[6] == 1 and pick == 4:
        run_mem.has_alice = True

    run_mem.update_target()
    return pick

if __name__ == "__main__":
    train_driver(episode_total = 1000)