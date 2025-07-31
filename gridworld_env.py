import numpy as np

class GridWorld():
  def __init__(self, goal_locations: list[tuple[int, int]], goal_rewards: list[int]):
    """A grid-based environment with an agent that can be stepped towards a goal. 
    
    Args: 
      goal_locations: List of locations of the goal(s)
      goal_rewards: List of rewards. The pair is respective to the goals location.
    """

    # List of goals
    self._goal_loc = goal_locations
    self._goal_rewards = goal_rewards

    if len(goal_locations) == 0: 
      raise ValueError("Need at least one goal location.")

    if len(goal_locations) != len(goal_rewards): 
      raise ValueError("The same number of goal locations and goal rewards should be specified.")
    
    # Build the GridWorld
    self._initialize_gridworld()

  def _initialize_gridworld(self):
    """
    GridWorld initialisation.
    
    ***Here, you should define the properties of GridWorld according to the figure in the handout,
    as well as the penalties/rewards associated with the features of GridWorld. Specifically, 
    _cliffs, _walls, _terminal_locs, _terminal_rewards, and _starting_loc. ***
    """

    # Properties of the GridWorld
    self._shape = (13, 10)
    
    # Location of cliffs
    self._cliffs = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
                    (4, 0), (4, 1), (4, 2), (4, 3),                         (4, 7), (4, 8), (4, 9),
                    (5, 0), (5, 1), (5, 2), (5, 3),                         (5, 7), (5, 8), (5, 9),
                    (6, 0), (6, 1), (6, 2), (6, 3),                         (6, 7), (6, 8), (6, 9),
                    (7, 0), (7, 1), (7, 2), (7, 3),
                    (8, 0), (8, 1), (8, 2), (8, 3)]

    # Location of wall
    self._walls = [(9, 0), (9, 1), (9, 2), (9, 3), (7, 7), (7, 8), (7, 9)]

    # Location of artefacts
    self._artefacts = [(2, 1), (9, 8)]

    # Denote locations that end an episode
    self._terminal_locs = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
                           (4, 0), (4, 1), (4, 2), (4, 3),                         (4, 7), (4, 8), (4, 9),
                                                   (5, 3),                         (5, 7),
                                                   (6, 3),                         (6, 7),
                                                   (7, 3),
                                                   (8, 3)] + self._goal_loc

    # Setup reward/penalties
    self._terminal_rewards = [-100 for _ in range(23)] + self._goal_rewards
    
    # Define a location for trials to start
    self._starting_loc = (11, 1)
    
    # Reward for standard tiles in GridWorld
    self._default_reward = 0
    
    # Maximum duration of each episode
    self._max_t = 500

    # Actions
    self._action_size = 4
    self._direction_names = ['N','E','S','W']

    # Sanity check for initial implmenetation. 
    if self._cliffs is None or self._walls is None or self._terminal_locs is None or self._terminal_rewards is None:
      raise NotImplementedError("Ensure all attributes are properly initialized.")
    
    # States
    self._locations = []
    for i in range (self._shape[0]):
      for j in range (self._shape[1]):
        loc = (i,j)
        # Adding the state to locations if it is no obstacle
        if self._is_location(loc):
          self._locations.append(loc)
    self._state_size = len(self._locations)

    # Set deterministic transitions
    self._probability_success_of_action = 1

    # Neighbours - each line is a state, ranked by state-number, each column is a direction (N, E, S, W)
    self._neighbours = np.zeros((self._state_size, 4))
    for state in range(self._state_size):
      loc = self.get_loc_from_state(state)

      neighbour = (loc[0]-1, loc[1])
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('N')] = self._get_state_from_loc(neighbour)
      else:
        self._neighbours[state][self._direction_names.index('N')] = state

      neighbour = (loc[0], loc[1]+1)
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('E')] = self._get_state_from_loc(neighbour)
      else:
        self._neighbours[state][self._direction_names.index('E')] = state

      neighbour = (loc[0]+1, loc[1])
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('S')] = self._get_state_from_loc(neighbour)
      else:
        self._neighbours[state][self._direction_names.index('S')] = state

      neighbour = (loc[0], loc[1]-1)
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('W')] = self._get_state_from_loc(neighbour)
      else:
        self._neighbours[state][self._direction_names.index('W')] = state

    # Absorbing condition to mark terminal locations that end an episode.
    self._absorbing = np.zeros((1, self._state_size), dtype=bool)
    for a in self._terminal_locs:
      absorbing_state = self._get_state_from_loc(a)
      self._absorbing[0, absorbing_state] = True
      
    # Transition matrix
    self._T = np.zeros((self._state_size, self._state_size, self._action_size))
    for action in range(self._action_size):
      for outcome in range(4):

        if action == outcome:
          prob = 1 - 3.0 * ((1.0 - self._probability_success_of_action) / 3.0)

        else:
          prob = (1.0 - self._probability_success_of_action) / 3.0

        # Write this probability in the transition matrix
        for prior_state in range(self._state_size):
          if not self._absorbing[0, prior_state]:
            post_state = self._neighbours[prior_state, outcome]
            post_state = int(post_state)
            self._T[prior_state, post_state, action] += prob

    # Reward matrix
    self._R = np.ones((self._state_size, self._state_size, self._action_size))
    self._R = self._default_reward * self._R
    for i in range(len(self._terminal_rewards)):
      post_state = self._get_state_from_loc(self._terminal_locs[i])
      self._R[:,post_state,:] = self._terminal_rewards[i]
      

    # Reset the environment output variables once inititialised
    self.reset()
    
    
  def get_cliffs_loc(self) -> list[tuple[int, int]]:
    """
    Get the location of cliffs. USED ONLY FOR PLOTTING!
    """
    return self._cliffs
  
  def get_walls_loc(self) -> list[tuple[int, int]]:
    """
    Get the location of walls. USED ONLY FOR PLOTTING!
    """
    return self._walls

  def get_gridshape(self) -> tuple[int, int]:
    """
    Get the shape of the gridworld. USED ONLY FOR PLOTTING!
    """
    return self._shape
  
  def get_starting_loc(self) -> tuple[int, int]:
    """
    Get starting location. USED ONLY FOR PLOTTING!
    """
    return self._starting_loc
  
  def get_goal_loc(self) -> list[tuple[int, int]]:
    """
    Get the location(s) of goal(s). USED ONLY FOR PLOTTING!
    """
    return self._goal_loc

  def _is_location(self, loc: tuple[int, int]) -> bool:
    """
    Check if the location is a valid state (not out of GridWorld and not a wall)

    Args: 
      loc: location of the state. 

    Returns: 
      Whether the location is a valid state.
    """
    row, col = loc

    # Check if location is within bounds
    if not (0 <= row < self._shape[0] and 0 <= col < self._shape[1]):
        return False

    # Check if location is a wall
    if loc in self._walls:
        return False

    return True

  def _get_state_from_loc(self, loc):
    """
    Get the state number corresponding to a given location
    input: loc {tuple} -- location of the state
    output: index {int} -- corresponding state number
    """
    return self._locations.index(tuple(loc))


  def get_loc_from_state(self, state):
    """
    Get the state number corresponding to a given location
    input: index {int} -- state number
    output: loc {tuple} -- corresponding location
    """
    return self._locations[state]

  def is_terminal(self, state: int) -> bool:
    """Returns if a state is terminal
    """
    return self._absorbing[0, state]

  def get_action_size(self) -> int:
    """Returns the number of actions. 
    """
    return self._action_size

  def get_state_size(self) -> int:
    """Returns the number of states.
    """
    return self._state_size

  # Functions used to perform episodes in the GridWorld environment
  def reset(self) -> tuple[int, int, bool, bool]:
    """
    Reset the environment state to starting state
    
    Returns: 
      - t, the current timestep
      - state, the current state of the envionment
      - reward, the current reward
      - done, True if reach a terminal state / False otherwise
    """
    self._t = 0
    self._state = self._get_state_from_loc(self._starting_loc)
    self._reward = 0
    self._done = False

    return self._t, self._state, self._reward, self._done


  def step(self, action: int) -> tuple[int, int, bool, bool]:
    """
    Perform an action in the environment

    Args: 
      action: action to perform

    Returns:
      - t, the current timestep
      - state, the current state of the envionment
      - reward, the current reward
      - done, True if reach a terminal state / False otherwise
    """

    # If environment already finished, print an error
    if self._done or self._absorbing[0, self._state]:
      print("Please reset the environment")
      return self._t, self._state, self._reward, self._done

    # Look for the first possible next states (so get a reachable state even if probability_success = 0)
    new_state = 0
    while self._T[self._state, new_state, action] == 0:
      new_state += 1
    assert self._T[self._state, new_state, action] != 0, "Selected initial state should be probability 0, something might be wrong in the environment."

    # Find the first state for which probability of occurence matches the random value
    total_probability = self._T[self._state, new_state, action]
    while (total_probability < self._probability_success_of_action) and (new_state < self._state_size-1):
     new_state += 1
     total_probability += self._T[self._state, new_state, action]
    assert self._T[self._state, new_state, action] != 0, "Selected state should be probability 0, something might be wrong in the environment."

    # Setting new t, state, reward and boolean 'done' condition
    self._t += 1
    self._reward = self._R[self._state, new_state, action]
    self._done = self._absorbing[0, new_state] or self._t > self._max_t
    self._state = new_state
    return self._t, self._state, self._reward, self._done
  