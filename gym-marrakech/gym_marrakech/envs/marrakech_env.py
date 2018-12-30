import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from .InvalidActionException import InvalidActionException

f = {
  "No": 0
  , "Ea": 1
  , "So": 2
  , "We": 3
}
dice = [1, 2, 2, 3, 3, 4]

def rollDie(np_random):
   return int(np_random.choice(dice))

def isOdd(num):
  return num & 0x1

# TODO: make for flexible board sizes
def goStep(position, facing):
  # left edge + go left
  if position[0] == 0 and facing == 3:
    if position[1] == 0:
      facing = 0
    elif isOdd(position[1]):
      position[1] += 1
      facing = 1
    else:
      position[1] -= 1
      facing = 1
    return (position, facing)

  # right edge + go right
  if position[0] == 6 and facing == 1:
    if position[1] == 6:
      facing = 2
    elif isOdd(position[1]):
      position[1] -= 1
      facing = 3
    else:
      position[1] += 1
      facing = 3
    return (position, facing)

  # bottom edge + go down
  if position[1] == 0 and facing == 2:
    if position[0] == 0:
      facing = 1
    elif isOdd(position[0]):
      position[0] += 1
      facing = 0
    else:
      position[0] -= 1
      facing = 0
    return (position, facing)

  # top edge + go up
  if position[1] == 6 and facing == 0:
    if position[0] == 6:
      facing = 3
    elif isOdd(position[0]):
      position[0] -= 1
      facing = 2
    else:
      position[0] += 1
      facing = 2
    return (position, facing)

  # else normal step - just go
  return goNormalStep(position, facing)


def goNormalStep(position, facing):
  if facing == 0:
    position += [0, 1]
  elif facing == 1:
    position += [1, 0]
  elif facing == 2:
    position += [0, -1]
  elif facing == 3:
    position += [-1, 0]
  return (position, facing)

def carpetPlacementIsInvalid(playerPosition, carpetPosition, orientation):
  # left edge + place left
  # right edge + place right
  # bottom edge + place down
  # top edge + place up
  if (playerPosition[0] == 0 and carpetPosition == 3) or \
    (playerPosition[0] == 6 and carpetPosition == 1) or \
    (playerPosition[1] == 0 and carpetPosition == 2) or \
    (playerPosition[1] == 6 and carpetPosition == 0):
    return True

  # left edge + place left direction
  # right edge + place right direction
  # bottom edge + place down direction
  # top edge + place up direction
  if (playerPosition[0] == 0 and carpetPosition == 0 and orientation == 0) or \
    (playerPosition[0] == 0 and carpetPosition == 2 and orientation == 2) or \
    (playerPosition[0] == 6 and carpetPosition == 0 and orientation == 2) or \
    (playerPosition[0] == 6 and carpetPosition == 2 and orientation == 0) or \
    (playerPosition[1] == 0 and carpetPosition == 3 and orientation == 0) or \
    (playerPosition[1] == 0 and carpetPosition == 1 and orientation == 2) or \
    (playerPosition[1] == 6 and carpetPosition == 3 and orientation == 2) or \
    (playerPosition[1] == 6 and carpetPosition == 1 and orientation == 0):
    return True

  # left + place left long
  # right + place right long
  # bottom + place down long
  # top + place up long
  if (playerPosition[0] == 1 and carpetPosition == 3 and orientation == 1) or \
    (playerPosition[0] == 5 and carpetPosition == 1 and orientation == 1) or \
    (playerPosition[1] == 1 and carpetPosition == 2 and orientation == 1) or \
    (playerPosition[1] == 5 and carpetPosition == 0 and orientation == 1):
    return True
  
  return False


class MarrakechEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  boardSize = (7, 7)
  numPlayers = 4
  numCarpets = 15

  def __init__(self):
    self.action_space = spaces.MultiDiscrete([
      3, # Movement: Left, Forward, Right
      4, # Carpet position: N, E, S, W of figure
      3  # Carpet orientation (looking from figure): left, straight, right 
    ])
    self.observation_space = spaces.Dict({
    	"board": spaces.Box(low=0, high=MarrakechEnv.numPlayers, shape=MarrakechEnv.boardSize, dtype=np.uint8)
      , "position": spaces.Box(low=np.zeros(2), high=np.array(MarrakechEnv.boardSize)-1, dtype=np.uint8)
      , "facing": spaces.Discrete(4) # N, E, S, W
      , "phase": spaces.MultiBinary(1) # 0 = move, 1 = lay down carpet
      , "round": spaces.Discrete(15) # every player has 15 carpets
    })
    self.seed()
    self.reset()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]


  def stepMoveAction(self, direction):
    # turn
    self.facing = (self.facing + (direction - 1)) % 4

    # move n steps
    leftSteps = numSteps = rollDie(self.np_random)
    while leftSteps:
      self.position, self.facing = goStep(self.position, self.facing)
      leftSteps -= 1

    return (0., {"numSteps": numSteps})

  def stepCarpetAction(self, cPosition, cOrientation):
    if carpetPlacementIsInvalid(self.position, cPosition, cOrientation):
      raise InvalidActionException()

    self.round += 1

    return (0., {})

  def step(self, action):
    assert self.action_space.contains(action)

    acMovement = action[0]
    acCarpetPos = action[1]
    acCarpetOri = action[2]

    backupObservation = self._getObs()

    result = (None, None, None, {"error": "Invalid Action"})
    try:
      if self.phase:
        reward, info = self.stepCarpetAction(acCarpetPos, acCarpetOri)
      else:
        reward, info = self.stepMoveAction(acMovement)

      # switch phase
      self.phase = 0 if self.phase else 1

      result = (
        self._getObs()
        , reward
        , self.gameIsOver()
        , info
      )
    except InvalidActionException:
      result = (
        backupObservation
        , 0.
        , self.gameIsOver()
        , {"error": "Invalid Action"}
      )
      pass

    return result

  def _getObs(self):
    return {
      "board": self.board,
      "position": self.position,
      "facing": self.facing,
      "phase": self.phase,
      "round": self.round
    }

  def _revertAction(self, backupObservation):
    self.board = backupObservation["board"]
    self.position = backupObservation["position"]
    self.facing = backupObservation["facing"]
    self.phase = backupObservation["phase"]
    self.round = backupObservation["round"]

  def gameIsOver(self):
    return self.round >= MarrakechEnv.numCarpets

  def reset(self):
    self.board = np.zeros(self.boardSize)
    self.position = np.array([
      np.floor(self.boardSize[0] / 2)
      , np.floor(self.boardSize[1] / 2)
    ], dtype=np.int8)
    self.facing = self.np_random.randint(4)
    self.round = 0
    self.phase = 0
    return self._getObs()

  def render(self, mode='human', close=False):
    print(' /\\/\\/\\/\\')
    for y in range(6, -1, -1):
      print('\\', end='') if isOdd(y) else print('/', end='')
      for x in range(7):
        if (self.position == [x,y]).all():
          if self.facing == 0:
            print('^', end='')
          elif self.facing == 1:
            print('>', end='')
          elif self.facing == 2:
            print('v', end='')
          elif self.facing == 3:
            print('<', end='')
        else:
          color = self.board[x,y]
          if color:
            print(chr(64 + color), end='')
          else:
            print('.', end='')
      print('\\') if isOdd(y) else print('/')
    print('\\/\\/\\/\\/')
