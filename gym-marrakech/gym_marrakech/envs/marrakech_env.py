import gym
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.pyplot as plt
import numpy as np
import io
from .InvalidActionException import InvalidActionException
from .RandomAi import RandomAi

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
def walkStep(position, facing):
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


def placeCarpet(board, playerPosition, color, carpetPosition, orientation):
  anchorPoint = playerPosition.copy()
  goNormalStep(anchorPoint, carpetPosition)
  board[tuple(anchorPoint)] = color
  orientation = (carpetPosition + (orientation - 1)) % 4
  goNormalStep(anchorPoint, orientation)
  board[tuple(anchorPoint)] = color



class MarrakechEnv(gym.Env):
  metadata = {'render.modes': ['human', 'ascii', 'rgb_array']}
  boardSize = (7, 7)
  numPlayers = 2
  numCarpets = 15

  def __init__(self, verbosity=0, players=None):
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
    self.action_space = spaces.Discrete(3)

    # merged together all information into a single vector
    # "board"    N x M flattened
    # "position" X, Y coordinate
    # "facing"   N, E, S, W
    # "phase"    0 = move, 1 = lay down carpet (removed)
    # "round"    every player has 15 carpets
    boardCells = MarrakechEnv.boardSize[0] * MarrakechEnv.boardSize[1]
    self.observation_space = spaces.Box(
      low = np.zeros(boardCells + 2 + 1 + 0 + 1),
      high = np.concatenate([
        np.ones(boardCells) * MarrakechEnv.numPlayers,
        np.array(MarrakechEnv.boardSize) - 1,
        np.array([3, 14])
      ]),
      dtype=np.uint8
    )

    if not players:
      self.otherPlayers = [RandomAi(), RandomAi(), RandomAi()]
    else:
      raise NotImplementedError("Can't use custom opponents yet.")

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
    self.currentAction = -1
    self.isPlayersMove = True
    self.currentPlayer = 0
    self.lastNumSteps = 0
    self.lastReward = 0.
    return self._getObs()

  def step(self, action):
    """
    if phase = move:
      use action to turn
      move x steps
      pay money
      if bankrupt:
        exit game
        done = True
    elif phase = carpet:
      lay carpet

    update counters
    if currentPlayer != 0:
      call other player to act
    """
    if self.verbosity & 0b100: print("do step for {} - {} {}".format(self.currentPlayer, self.round, self.phase))
    assert self.action_space.contains(action)

    if self.gameIsOver():
      return (self._getObs(), 0, True, {"error": "Game Over"})

    acMovement = action
    self.currentAction = action

    reward = 0.
    info = {}

    rewards, info = self._stepMoveAction(acMovement)
    reward = rewards[self.currentPlayer]


    acCarpetPos = self.np_random.randint(4)
    acCarpetOri = self.np_random.randint(3)
    #print("carpet (%d %d)" % (acCarpetPos, acCarpetOri))
    self._stepCarpetAction(acCarpetPos, acCarpetOri)
    self.currentPlayer += 1

    if self.currentPlayer >= MarrakechEnv.numPlayers:
      self.currentPlayer = 0
      self.round += 1

    if self.isPlayersMove:
      self.lastBalance = self.accounts[0]
      self.isPlayersMove = False
      while self.currentPlayer != 0:
        #print("self._triggerOtherPlayer()")
        if self._triggerOtherPlayer():
          break
      self.isPlayersMove = True
      reward += self.accounts[0] - self.lastBalance

    return (
      self._getObs()
      , reward
      , self.gameIsOver()
      , info
    )

  def _splitAction(self, action):
    acMovement = (action & 0b11) % 3
    acCarpetPos = ((action >> 2) & 0b11) % 4
    acCarpetOri = ((action >> 4) & 0b11) % 3
    return (acMovement, acCarpetPos, acCarpetOri)

  def _stepMoveAction(self, direction):
    # turn
    self.facing = (self.facing + (direction - 1)) % 4

    # move x steps
    leftSteps = numSteps = rollDie(self.np_random)
    #print("numsteps " + str(numSteps))
    while leftSteps:
      self.position, self.facing = walkStep(self.position, self.facing)
      leftSteps -= 1

    self.lastNumSteps = numSteps

    # pay money
    rewards = [0., 0., 0., 0.]
    steppedColor = self._getColorAtPosition()
    steppedPlayer = steppedColor - 1
    if steppedColor > 0 and self.currentPlayer != steppedPlayer:
      connectedTiles = len(self._getConnectedCarpetsFromPosition())
      moneyTransferAmount = min(connectedTiles, self.accounts[self.currentPlayer])
      self.accounts[self.currentPlayer] -= moneyTransferAmount
      self.accounts[steppedPlayer] += moneyTransferAmount
      rewards[self.currentPlayer] -= moneyTransferAmount
      rewards[steppedPlayer] += moneyTransferAmount

    return (rewards, {"numSteps": numSteps})

  def _getColorAtPosition(self):
    return self.board[tuple(self.position)]

  def _getConnectedCarpetsFromPosition(self):
    colorAtPos = self._getColorAtPosition()
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

  def _stepCarpetAction(self, cPosition, cOrientation):
    reward = 0.
    if carpetPlacementIsInvalid(self.position, cPosition, cOrientation):
      reward = 0.
    else:
      placeCarpet(self.board, self.position, self.currentPlayer + 1, cPosition, cOrientation)
    return (reward, {})

  def _getObs(self):
    return np.concatenate([
      self.board.flatten(),
      self.position,
      np.array([self.facing, self.round])
    ])

  def _triggerOtherPlayer(self):
    if self.verbosity & 0b10:
      display(self.render())

    otherPlayer = self.otherPlayers[self.currentPlayer-1]
    action = otherPlayer.requestAction(self)
    #print("other player: move " + str(action))
    observation, reward, done, info = self.step(action)

    return done

  def gameIsOver(self):
    return self.round >= MarrakechEnv.numCarpets or self.accounts[0] <= 0

  def render(self, mode='human', close=False, prefix=""):
    if mode == 'ascii':
      return self.renderAscii(prefix=prefix)
    fig = self.renderImg()
    if mode == 'rgb_array':
      fig.canvas.draw()
      return np.array(fig.canvas.renderer._renderer)
    else:
      return fig

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
    fig = plt.figure(figsize=np.array(fullPic.shape)[::-1]/20, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    plt.axis('off')
    plt.imshow(fullPic, cmap=cmap, norm=norm, aspect='equal', origin='lower')
    #plt.imshow(np.rot90(fullPic), cmap=cmap, norm=norm, aspect='equal')

    action = self.currentAction
    acMovement = (action & 0b11) % 3
    acCarpetPos = ((action >> 2) & 0b11) % 4
    acCarpetOri = ((action >> 4) & 0b11) % 3

    info = "round %d  player %d phase %d\n" % (self.round, self.currentPlayer, self.phase)
    if self.currentAction >= 0:
      if self.phase: # inverse, because phase change already happened
        info += "last ACT - Move: {} steps {}".format(self.lastNumSteps, orientationSpaceShort[acMovement])
      else:
        info += "last ACT - Carpet: ({}, {})".format(directionSpaceShort[acCarpetPos], orientationSpaceShort[acCarpetOri])
    info += "\n%s %.1f " % (playerColor[self._getColorAtPosition()], self.lastReward)
    info += ','.join([str(x) for x in self.accounts])
    plt.text(3, cellSize*7 + 2*cellSize/3, info, fontsize=10)
    plt.close()
    return fig


import matplotlib as mpl
from matplotlib import colors

cellSize = 9
playerSize = int(cellSize/3)
borders = ((int(cellSize/3), int(1.7*cellSize + cellSize/3)), (int(cellSize/3), int(cellSize/3)))
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
