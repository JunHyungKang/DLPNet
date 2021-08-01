import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.optim import Adam
import numpy as np
import torchvision.models as torchmodels


def set_parameter_requires_grad(model, feature_extracting):
    # When loading the models, make sure to call this function to update the weights
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def resnet_model(num_action):
    agent = torchmodels.resnet18(pretrained=True)
    set_parameter_requires_grad(agent, False)
    num_ftrs = agent.fc.in_features
    agent.fc = torch.nn.Linear(num_ftrs, num_action)
    return agent


class ReinforceDisc(object):
    def __init__(self, num_action):
        self.num_action = num_action
        self.agent = resnet_model(num_action)
        self.action_space = [(1.6, 4.4), (1.8, 4.2), (2, 4), (2.2, 3.8), (2.4, 3.6)]

    def get_policy(self, data):
        logits = self.agent(data)
        return Categorical(logits=logits)

    def get_action(self, obs):
        sample = self.get_policy(obs).sample()
        return sample

    def get_parameter(self, action):
        return [self.action_space[x] for x in action]

    def compute_logp(self, obs, act):
        logp = self.get_policy(obs).log_prob(act)
        return logp

    def compute_loss(self, logp, weights):
        return -(logp * weights).mean()

    def reward_to_go(self, rews):
        n = len(rews)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0)
        return rtgs


class PPO(object):
    def __init__(self, num_action):
        self.num_action = num_action
        self.pi = resnet_model(num_action)
        self.v = resnet_model(1)
        self.action_space = [(1.6, 4.4), (1.8, 4.2), (2, 4), (2.2, 3.8), (2.4, 3.6)]

    def get_policy(self, data):
        logits = self.agent(data)
        return Categorical(logits=logits)

    def get_action(self, obs):
        sample = self.get_policy(obs).sample()
        return sample

    def get_parameter(self, action):
        return [self.action_space[x] for x in action]

    def compute_logp(self, obs, act):
        logp = self.get_policy(obs).log_prob(act)
        return logp

    def compute_loss(self, logp, weights):
        return -(logp * weights).mean()

    def reward_to_go(self, rews):
        n = len(rews)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + 0.95 * (rtgs[i + 1] if i + 1 < n else 0)
        return rtgs






