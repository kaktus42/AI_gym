from numpy import random

class RandomAi(object):
  def step(self, observation, env):
    try:
      return random.choice(env.getValidActions())
    except AttributeError:
      return env.action_space.sample()
