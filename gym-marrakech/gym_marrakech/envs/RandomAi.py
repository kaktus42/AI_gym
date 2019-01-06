from numpy import random

class RandomAi(object):
  def requestAction(self, env):
    try:
      return random.choice(env.getValidActions())
    except AttributeError:
      return env.action_space.sample()
