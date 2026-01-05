import gymnasium as gym
from gymnasium.envs.registration import register
from minigrid.envs.multiroom import MultiRoomEnv
from minigrid.core.world_object import Door, Key, Lava, Goal, Wall
from minigrid.core.grid import Grid
import random
import time
from collections import deque

""" Script: Visualization of Custom MultiColor Corridor Environment with Lava in MiniGrid
This script visualizes a custom MiniGrid environment where the agent must navigate through multiple rooms,
collecting keys of different colors to unlock corresponding doors, while avoiding lava obstacles.
It combines elements from all the trainings done previously.
"""

class LavaCorridorFinal(MultiRoomEnv):
    def __init__(self, n_rooms=4, key_prob=0.2, lava_prob=0.25, **kwargs):
        self.target_n_rooms = n_rooms   # Desired number of rooms
        self.key_prob = key_prob  # Probability of placing a key-door pair in a room
        self.lava_prob = lava_prob  # Probability of placing lava in non-safe cells
        
        super().__init__(
            minNumRooms=n_rooms, # Minimum number of rooms
            maxNumRooms=n_rooms,  # Maximum number of rooms
            maxRoomSize=10, # Maximum room size
            **kwargs
        )

    def _gen_grid(self, width, height):
        max_retries = 500
        for _ in range(max_retries):
            
            # 1. Basic Grid Generation
            self.grid = Grid(width, height)
            try:
                super()._gen_grid(width, height)
            except Exception:
                continue 
            
            # 2. Place Key-Door Pairs
            valid_colors = ['red', 'blue', 'purple', 'yellow', 'grey']
            for i, room in enumerate(self.rooms):
                if i == len(self.rooms) - 1: break 
                
                if random.random() < self.key_prob:
                    door_pos = room.exitDoorPos
                    color = random.choice(valid_colors)
                    self.grid.set(door_pos[0], door_pos[1], Door(color, is_locked=True))
                    # Colocamos la llave
                    self.place_obj(Key(color), top=room.top, size=room.size, max_tries=100)

            # 3. Add Lava Smartly, protecting Keys and Doors
            self._add_lava_smart()

            # 4. Final Validation (BFS)
            if self._is_path_clear():
                return 

        print("⚠️ Could not generate a solvable level. Reducing lava.")
        self.lava_prob = 0.05
        super()._gen_grid(width, height)

    def _add_lava_smart(self):
        """
        Adds lava to the grid while ensuring that keys and doors remain accessible.
        The method identifies 'safe' cells around critical objects (keys, doors, goal) and only places lava in non-safe cells.
        """
        safe_cells = set()
        safe_cells.add(self.agent_pos)

        # A) Identify safe cells around Keys, Doors, and Goal
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                obj = self.grid.get(x, y)
                
                # If it's a Key, Door, or Goal -> It's safe
                if isinstance(obj, (Key, Door, Goal)):
                    safe_cells.add((x, y))
                    # Add its 4 neighbors to the safe list (Breathing room)
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        safe_cells.add((x + dx, y + dy))

        # B) Also add door positions stored in Room objects
        # (Sometimes Minigrid deletes the door object when merging, but the position is key)
        for room in self.rooms:
            if room.entryDoorPos:
                safe_cells.add(room.entryDoorPos)
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    safe_cells.add((room.entryDoorPos[0]+dx, room.entryDoorPos[1]+dy))
            
            if room.exitDoorPos:
                safe_cells.add(room.exitDoorPos)
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    safe_cells.add((room.exitDoorPos[0]+dx, room.exitDoorPos[1]+dy))

        # C) Place Lava in non-safe cells
        for room in self.rooms:
            for x in range(room.top[0], room.top[0] + room.size[0]):
                for y in range(room.top[1], room.top[1] + room.size[1]):
                    
                    # Check that the cell is empty of objects
                    cell = self.grid.get(x, y)
                    
                    # If the cell is empty AND NOT in the safe list, consider placing lava
                    if cell is None and (x, y) not in safe_cells:
                        if random.random() < self.lava_prob:
                            self.grid.set(x, y, Lava())

    def _is_path_clear(self):
        # BFS to check if there's a path from agent to goal
        start = self.agent_pos
        end = None
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                if isinstance(self.grid.get(i, j), Goal):
                    end = (i, j)
                    break
        if not end: return False 

        queue = deque([start])
        visited = set([start])

        while queue:
            x, y = queue.popleft()
            if (x, y) == end: return True

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                    if (nx, ny) not in visited:
                        cell = self.grid.get(nx, ny)
                        # Caminable: Vacío, Meta, Llave, Puerta (abierta o cerrada)
                        if not isinstance(cell, (Lava, Wall)):
                            visited.add((nx, ny))
                            queue.append((nx, ny))
        return False

# Register the custom environment
if "MiniGrid-VisualizerFinal-v0" in gym.envs.registry:
    del gym.envs.registry["MiniGrid-VisualizerFinal-v0"]

register(
    id="MiniGrid-VisualizerFinal-v0",
    entry_point=__name__ + ":LavaCorridorFinal",
)

# Main function to visualize the environment
def main():

    N_ROOMS = 12 # Number of rooms
    LAVA_PROBABILITY = 0.25 # Probability of lava placement
        
    env = gym.make( # Create the environment with the parameters
        "MiniGrid-VisualizerFinal-v0", 
        render_mode="human", 
        n_rooms=N_ROOMS,
        lava_prob=LAVA_PROBABILITY
    )

    # Generate 3 different games
    for i in range(3):
        env.reset()
        for _ in range(100): 
            env.render()
            time.sleep(0.1)
    env.close()

if __name__ == "__main__":
    main()