import gym
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.pyplot as plt
import numpy as np
import io
from .InvalidActionException import InvalidActionException

dice = [1, 2, 2, 3, 3, 4]

orientationSpace = ["left", "straight", "right"]
orientationSpaceShort = ["le", "str", "ri"]
directionSpace = ["North", "East", "South", "West"]
directionSpaceShort = ["N", "E", "S", "W"]
playerColor = ["None", "Blue", "Yellow", "Red", "Green"]

def printAction(action, prefix=""):
    acMovement = (action & 0b11) % 3
    acCarpetPos = ((action >> 2) & 0b11) % 4
    acCarpetOri = ((action >> 4) & 0b11) % 3
    print("{}ACTION Move: {}, Carpet: ({}, {})".format(prefix, orientationSpace[acMovement], directionSpace[acCarpetPos], orientationSpace[acCarpetOri]))
    
def printObservation(observation, prefix=""):
    print("{}Position/Facing: {} {}".format(prefix, observation[-5:-3], directionSpace[int(observation[-3:-2][0])]))
    print("%sPhase: %d" % (prefix, int(observation[-2:-1][0])))
    print("%sRound: %d" % (prefix, int(observation[-1:][0])))

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


def placeCarpet(board, playerPosition, playerNumber, carpetPosition, orientation):
  anchorPoint = playerPosition.copy()
  goNormalStep(anchorPoint, carpetPosition)
  board[tuple(anchorPoint)] = playerNumber
  orientation = (carpetPosition + (orientation - 1)) % 4
  goNormalStep(anchorPoint, orientation)
  board[tuple(anchorPoint)] = playerNumber



class MarrakechEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  boardSize = (7, 7)
  numPlayers = 4
  numCarpets = 15

  def __init__(self, verbosity=0):
    #self.action_space = spaces.MultiDiscrete([
    #  3, # Movement: Left, Forward, Right
    #  4, # Carpet position: N, E, S, W of figure
    #  3  # Carpet orientation (looking from figure): left, straight, right 
    #])
    #self.observation_space = spaces.Dict({
    #  "board": spaces.Box(low=0, high=MarrakechEnv.numPlayers, shape=MarrakechEnv.boardSize, dtype=np.uint8)
    #  , "position": spaces.Box(low=np.zeros(2), high=np.array(MarrakechEnv.boardSize)-1, dtype=np.uint8)
    #  , "facing": spaces.Discrete(4) # N, E, S, W
    #  , "phase": spaces.MultiBinary(1) # 0 = move, 1 = lay down carpet
    #  , "round": spaces.Discrete(15) # every player has 15 carpets
    #})
    
    # binary representation: xx yy zz
    # x = Carpet orientation (0-2)
    # y = Carpet position (0-3)
    # z = Movement (0-2)
    self.action_space = spaces.Discrete(47)

    # merged together all information into a single vector
    # "board"    N x M flattened
    # "position" X, Y coordinate
    # "facing"   N, E, S, W
    # "phase"    0 = move, 1 = lay down carpet
    # "round"    every player has 15 carpets
    boardCells = MarrakechEnv.boardSize[0] * MarrakechEnv.boardSize[1]
    self.observation_space = spaces.Box(
      low = np.zeros(boardCells + 2 + 1 + 1 + 1),
      high = np.concatenate([
        np.ones(boardCells) * MarrakechEnv.numPlayers,
        np.array(MarrakechEnv.boardSize) - 1,
        np.array([3, 1, 14])
      ]),
      dtype=np.uint8
    )

    self.verbosity = verbosity
    self.seed()
    self.reset()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset(self):
    self.board = np.zeros(self.boardSize, dtype=np.int8)
    self.position = np.array([
      np.floor(self.boardSize[0] / 2)
      , np.floor(self.boardSize[1] / 2)
    ], dtype=np.int8)
    self.facing = self.np_random.randint(4)
    self.round = 0
    self.phase = 0
    self.accounts = [30, 30, 30, 30]
    self.lastAction = -1
    self.isPlayersMove = True
    self.playerNumber = 1
    self.lastNumSteps = 0
    self.lastReward = 0.
    return self._getObs()

  def stepMoveAction(self, direction):
    # turn
    self.facing = (self.facing + (direction - 1)) % 4

    # move n steps
    leftSteps = numSteps = rollDie(self.np_random)
    while leftSteps:
      self.position, self.facing = goStep(self.position, self.facing)
      leftSteps -= 1

    self.lastNumSteps = numSteps

    reward = 0.
    steppedOnColor = self.getColorAtPosition()
    if steppedOnColor > 0 and self.playerNumber != steppedOnColor:
      connectedTiles = len(self.getConnectedCarpetsFromPosition())
      self.accounts[self.playerNumber-1] -= connectedTiles
      self.accounts[steppedOnColor-1] += connectedTiles
      reward -= connectedTiles

    return (reward, {"numSteps": numSteps})

  def stepCarpetAction(self, cPosition, cOrientation):
    reward = 0.
    if carpetPlacementIsInvalid(self.position, cPosition, cOrientation):
      reward = -100.
    else:
      placeCarpet(self.board, self.position, self.playerNumber, cPosition, cOrientation)

    if self.isPlayersMove:
      self.round += 1

    self.lastNumSteps = 0

    return (reward, {})

  def step(self, action):
    assert self.action_space.contains(action)

    acMovement = (action & 0b11) % 3
    acCarpetPos = ((action >> 2) & 0b11) % 4
    acCarpetOri = ((action >> 4) & 0b11) % 3
    self.lastAction = action

    backupObservation = self._getObs()

    result = (None, None, None, {"error": "Invalid Action"})
    try:
      if self.phase:
        reward, info = self.stepCarpetAction(acCarpetPos, acCarpetOri)
        self.lastReward = reward
        self.phase = 0

        if self.isPlayersMove:
          self.isPlayersMove = False
          reward += self.makePlayersSteps()
          self.isPlayersMove = True
          self.playerNumber = 1
        self.lastReward = reward
      else:
        reward, info = self.stepMoveAction(acMovement)
        self.lastReward = reward
        self.phase = 1

      result = (
        self._getObs()
        , reward
        , self.gameIsOver()
        , info
      )
    except InvalidActionException:
      result = (
        backupObservation
        , -100.
        , self.gameIsOver()
        , {"error": "Invalid Action"}
      )
      pass

    return result

  def _getObs(self):
    return np.concatenate([
      self.board.flatten(),
      self.position,
      np.array([self.facing, self.phase, self.round])
    ])

  def getColorAtPosition(self):
    return self.board[tuple(self.position)]

  def getConnectedCarpetsFromPosition(self):
    colorAtPos = self.getColorAtPosition()
    newFields = set([tuple(self.position)])
    fields = set()

    while fields != newFields:
        fields = newFields.copy()
        for f in fields:
            for o in [(1,0), (-1,0), (0,1), (0,-1)]:
                testPos = (f[0] + o[0], f[1] + o[1])
                try:
                    if colorAtPos == self.board[testPos]:
                        newFields.add(testPos)
                except IndexError:
                    pass
    return newFields

  def _revertAction(self, backupObservation):
    self.board = backupObservation["board"]
    self.position = backupObservation["position"]
    self.facing = backupObservation["facing"]
    self.phase = backupObservation["phase"]
    self.round = backupObservation["round"]

  def makePlayersSteps(self):
    totalReturnedReward = 0
    for self.playerNumber in np.arange(2, MarrakechEnv.numPlayers + 1):
      if self.verbosity & 0b100:
        self.render(prefix="#%d# " % self.playerNumber)
      if self.verbosity & 0b1:
        print("#################")
        print("### PLAYER %d ###" % self.playerNumber)
        print("#################")
      action = self.action_space.sample()
      if self.verbosity & 0b1:
        printAction(action, "#%d# " % self.playerNumber)
      observation, reward, done, info = self.step(action)
      if self.getColorAtPosition() == 1:
        totalReturnedReward = -reward
      if self.verbosity & 0b1:
        print("#%d# %d steps, %.1f reward" % (self.playerNumber, info["numSteps"], reward))

      if self.verbosity & 0b100:
        self.render(prefix="#%d# " % self.playerNumber)
      action = self.action_space.sample()
      if self.verbosity & 0b1:
        printAction(action, "#%d# " % self.playerNumber)
      observation, reward, done, info = self.step(action)
      if self.verbosity & 0b1:
        print("#%d# %.1f reward" % (self.playerNumber, reward))
      if self.verbosity & 0b10:
        self.render(prefix="#%d# " % self.playerNumber)

    return totalReturnedReward

  def gameIsOver(self):
    return self.round >= MarrakechEnv.numCarpets

  def render(self, mode='human', close=False, prefix=""):
    if mode == 'ascii':
      return self.renderAscii(prefix=prefix)
    else:
      return self.renderImg()

  def renderAscii(self, prefix=""):
    buf = io.StringIO()
    buf.write(prefix + ' /\\/\\/\\/\\\n')
    for y in range(6, -1, -1):
      buf.write(prefix + ('\\' if isOdd(y) else '/'))
      for x in range(7):
        if (self.position == [x,y]).all():
          if self.facing == 0:
            buf.write('^')
          elif self.facing == 1:
            buf.write('>')
          elif self.facing == 2:
            buf.write('v')
          elif self.facing == 3:
            buf.write('<')
        else:
          color = self.board[x,y]
          buf.write(chr(64 + color) if color else '.')
      buf.write('\\' if isOdd(y) else '/')
      buf.write('\n')
    buf.write(prefix + '\\/\\/\\/\\/\n')
    return buf.getvalue()

  def renderImg(self):
    fullPic = np.zeros((cellSize*7 + borders[0][0] + borders[0][1], cellSize*7 + borders[1][0] + borders[1][1]))
    for y in range(6, -1, -1):
      for x in range(7):
        color = self.board[x,y]
        drawBlockOnPic(fullPic, x, y, color, self.position, self.facing)
    fig = plt.figure()
    plt.axis('off')
    plt.imshow(fullPic, cmap=cmap, norm=norm, aspect='equal', origin='lower')
    #plt.imshow(np.rot90(fullPic), cmap=cmap, norm=norm, aspect='equal')

    action = self.lastAction
    acMovement = (action & 0b11) % 3
    acCarpetPos = ((action >> 2) & 0b11) % 4
    acCarpetOri = ((action >> 4) & 0b11) % 3

    info = "round %d  player %d phase %d\n" % (self.round, self.playerNumber, self.phase)
    if self.lastAction >= 0:
      if self.phase: # inverse, because phase change already happened
        info += "ACT Move: {} steps {}".format(self.lastNumSteps, orientationSpaceShort[acMovement])
      else:
        info += "ACT Carpet: ({}, {})".format(directionSpaceShort[acCarpetPos], orientationSpaceShort[acCarpetOri])
    info += "\n%s %.1f " % (playerColor[self.getColorAtPosition()], self.lastReward)
    info += ','.join([str(x) for x in self.accounts])
    plt.text(1, 9*7 + 7, info, fontsize=10)
    return fig


import matplotlib as mpl
from matplotlib import colors

cellSize = 9
playerSize = int(cellSize/3)
borders = ((int(cellSize/3), int(cellSize/3)), (int(cellSize + cellSize/3), int(cellSize/3)))
playerMask = np.array(
    [1 if x < cellSize/2 and (abs(int(x-(cellSize/2)+0.5))+abs(int(y-(cellSize/2)+0.5))) < cellSize*0.4 else 0 for x in range(cellSize) for y in range(cellSize)]
).reshape((cellSize,cellSize))
cmap = colors.ListedColormap(['#F9E4B7', 'blue', 'yellow', 'red', 'green', 'white', 'black'])
bounds = [0, 1, 2, 3, 4, 5, 6, 7]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
rawBoardCell = np.zeros((cellSize, cellSize), dtype=int)
rawBoardCell[1:-1, 1:-1] = 1

def drawBlockOnPic(fullPic, y, x, col, playerPos, playerOri):
    offX = cellSize * x + borders[0][0]
    offY = cellSize * y + borders[1][0]
    newCell = rawBoardCell.copy() * col
    if (playerPos == [y,x]).all():
      newCell = (newCell & ~playerMask) + playerMask*6
      newCell = np.rot90(newCell, playerOri-2)
    newCell += (rawBoardCell == 0) * 5
    fullPic[offX:offX+rawBoardCell.shape[0], offY:offY+rawBoardCell.shape[1]] = newCell
