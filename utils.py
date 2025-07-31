import matplotlib.pyplot as plt
import numpy as np
from gridworld_env import GridWorld

def paint_gridworld(gridworld: GridWorld, 
                    figsize: tuple[int, int]=(17, 12), 
                    actions: np.ndarray = None,
                    ticksize: int=15):
  """
  Paint the gridworld.

  Args: 
    gridworld_obj: GridWorld Object
    figsize: Figure size. 
    actions: Array of integers that matches 
        gridworld_obj.get_gridshape(). Values not in range 
        [0, 3] will not be plotted. 
    ticksize: Size of the ticks. 
  """
  grid = _get_gridworld_as_array(gridworld)
  
  plt.figure(figsize=figsize)
  plt.imshow(grid)
  plt.xticks(fontsize=ticksize)
  plt.yticks(fontsize=ticksize)

  if actions is not None:
      if actions.shape != gridworld.get_gridshape():
         raise ValueError("Action shape must match gridworld shape.")
      
      _plot_arrows(actions)

  plt.show()

def _plot_arrows(actions: np.ndarray, arrow_size: float=0.3):
    """
    Plots arrows on the grid based on the provided actions.

    Assumes a mapping of 0 (North), 1 (East), 2 (South), 3 (West)
    """
    shape = actions.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            action = actions[i, j]
            if action == 2:  # Up
                dx, dy = 0, arrow_size
            elif action == 1:  # Right
                dx, dy = arrow_size, 0
            elif action == 0:  # Down
                dx, dy = 0, -arrow_size
            elif action == 3:  # Left
                dx, dy = -arrow_size, 0
            else:
                continue  # No action or invalid action

            plt.arrow(j, i, dx, dy, 
                      head_width=0.2, 
                      head_length=0.2, 
                      fc='black', ec='black')



def _get_gridworld_as_array(gridworld_obj):
  """
  Return the grid with edges location, goal locations, starting locations, and walls
  This can be directly passed in to paint_gridworld to paint the map
  """
  shape = gridworld_obj.get_gridshape()
  edges_locs = gridworld_obj.get_cliffs_loc()
  goal_states = gridworld_obj.get_goal_loc()
  starting_loc = gridworld_obj.get_starting_loc()
  walls = gridworld_obj.get_walls_loc()

  grid = np.ones((shape[0], shape[1], 3), dtype=int)
  grid[:, :, :] = [255, 255, 255]
  
  for ob in edges_locs:
    grid[ob] = [30,144,255]

  for goal in goal_states:
    grid[goal] = [178,34,34]

  for wall in walls:
      grid[wall] = [0,0,0]

  grid[starting_loc] = [253, 218, 13]
  return grid