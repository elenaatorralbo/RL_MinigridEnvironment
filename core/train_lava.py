import gymnasium as gym
from gymnasium import spaces, ObservationWrapper
from gymnasium.envs.registration import register
from minigrid.envs.multiroom import MultiRoomEnv
from minigrid.core.world_object import Door, Key, Lava, Wall, Goal
from minigrid.core.grid import Grid
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import random
import os
import collections

# =============================================================================
# 1. CUSTOM MULTICOLOR CORRIDOR WITH SMART LAVA ENVIRONMENT
# Environment with several rooms, doors of different colors with corresponding keys,
# and lava obstacles placed intelligently to avoid blocking critical paths.
# =============================================================================
class CorredorLavaSmart(MultiRoomEnv):
    def __init__(self, n_rooms=4, lava_prob=0.1, key_prob=0.1, **kwargs): # Initialize with number of rooms, lava and key probabilities
        super().__init__(
            minNumRooms=n_rooms, # Minimum number of rooms
            maxNumRooms=n_rooms, # Maximum number of rooms
            maxRoomSize=8, # Maximum room size, this dimensions consider the walls, so the real dimensions of the room are (maxRoomSize-2) x (maxRoomSize-2)
            **kwargs
        )
        self.key_prob = key_prob
        self.lava_prob = lava_prob

    def _gen_grid(self, width, height):
        # Try multiple times to generate a valid level
        max_retries = 500
        for _ in range(max_retries):
            
            # 1.Basic Grid Generation
            self.grid = Grid(width, height)
            try:
                super()._gen_grid(width, height)
            except Exception:
                continue

            # 2. Place Key-Door Pairs Probabilistically in all rooms
            self._place_doors_probabilistic()

            # 3. Add Lava Smartly, protecting Keys and Doors
            self._add_lava_smart()

            # 4. Path Validation (BFS)
            if self._is_path_clear():
                return 

        # If all attempts fail, generate one without lava to avoid crash
        print("Could not generate a solvable level. Generating without lava.")
        self.lava_prob = 0.0
        super()._gen_grid(width, height)

    # Place doors and keys based on probability
    def _place_doors_probabilistic(self):
        valid_colors = ['red', 'blue', 'purple', 'yellow', 'grey']
        
        for i, room in enumerate(self.rooms):

            if i == len(self.rooms) - 1: # No door in the last room
                break
            
            if random.random() < self.key_prob: # Decide to place a door-key pair based on probability
                door_pos = room.exitDoorPos
                color = random.choice(valid_colors)
                
                self.grid.set(door_pos[0], door_pos[1], Door(color, is_locked=True)) # Place the door
                
                self.place_obj(Key(color), top=room.top, size=room.size, max_tries=100) # Place the corresponding key


    """ 
    Add lava obstacles while ensuring critical items and paths remain accessible. It protects:
        A) Keys, Doors, and Goal by marking their cells and adjacent cells as safe.
        B) Entry and Exit doors of rooms to maintain structural connectivity. 
    """
    def _add_lava_smart(self): 

        safe_cells = set()
        safe_cells.add(self.agent_pos)

        # A) Search for Keys, Doors, and Goal
        # Scans the entire grid, so it will detect if there is 1 key or 5 keys.
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                obj = self.grid.get(x, y)
                if isinstance(obj, (Key, Door, Goal)):
                    safe_cells.add((x, y))

                    # Also mark adjacent cells as safe
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        safe_cells.add((x + dx, y + dy))

        # B) Ensure entry/exit doors of rooms (structural gaps)
        for room in self.rooms:
            for d_pos in [room.entryDoorPos, room.exitDoorPos]:
                if d_pos:
                    safe_cells.add(d_pos)
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        safe_cells.add((d_pos[0]+dx, d_pos[1]+dy))

        # C) Place Lava only in non-safe cells
        for room in self.rooms:

            # Iterate through each cell in the room
            for x in range(room.top[0], room.top[0] + room.size[0]):
                for y in range(room.top[1], room.top[1] + room.size[1]):
                    
                    cell = self.grid.get(x, y)
                    
                    # Place lava if the cell is empty and not marked as safe
                    if cell is None and (x, y) not in safe_cells:
                        if random.random() < self.lava_prob:
                            self.grid.set(x, y, Lava())

    """ 
    Ensure there is a valid path from the agent's start position to the goal, avoiding lava and walls. 
    It uses a simple Breadth-First Search (BFS) algorithm for pathfinding. 
    """
    def _is_path_clear(self): # Define a simple BFS to check connectivity

        start = self.agent_pos
        end = None
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if isinstance(self.grid.get(x, y), Goal):
                    end = (x, y)
                    break
        if not end: return False 

        queue = collections.deque([start])
        visited = set([start])

        while queue:
            x, y = queue.popleft()
            if (x, y) == end: return True

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                    if (nx, ny) not in visited:
                        cell = self.grid.get(nx, ny)
                        if not isinstance(cell, (Lava, Wall)):
                            visited.add((nx, ny))
                            queue.append((nx, ny))
        return False
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if terminated and reward <= 0:
            reward = -0.5  
            
        return obs, reward, terminated, truncated, info

# Register the custom environment
if "MiniGrid-LavaSmartMulti-v0" in gym.envs.registry:
    del gym.envs.registry["MiniGrid-LavaSmartMulti-v0"]

register(
    id="MiniGrid-LavaSmartMulti-v0",
    entry_point=__name__ + ":CorredorLavaSmart",
)

# =============================================================================
# 2. IMAGE OBSERVATION WRAPPER: to use only preprocessed image observations. This wrapper extracts only 
# the 'image' tensor (H, W, C) to make it compatible with the Stable-Baselines3 CNN/Mlp policies.
# =============================================================================
class ImgObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        img_space = env.observation_space.spaces["image"]   # Extract 'image' space
        self.observation_space = spaces.Box(    # Define new observation space as a Box of pixel values ("uint8": 0-255)
            low=0,
            high=255,
            shape=img_space.shape,
            dtype="uint8"
        )

    def observation(self, obs): # Return only the image part of the observation
        return obs["image"]

# =============================================================================
# 3. GRADUAL LAVA + MULTIKEY CURRICULUM TRAINING FUNCTION
# =============================================================================
def train_lava_smart_multi():
    
    initial_model_path = "Fase_4_Color_12Hab_FINAL4.zip" # Pre-trained model path
    
    # Curriculum Stages Configuration
    stages_config = [
        {"rooms": 1, "lava": 0.15, "key": 0.0, "steps": 1_000_000}, # Phase 1: Simple navigation with lava
        {"rooms": 3, "lava": 0.05, "key": 0.10, "steps": 2_000_000}, # Phase 2: Introduce keys and doors
        {"rooms": 6, "lava": 0.075, "key": 0.15, "steps": 2_000_000}, # Phase 3: More rooms with higher lava density
        {"rooms": 9, "lava": 0.10, "key": 0.20, "steps": 2_000_000}, # Phase 4: Increased complexity
        {"rooms": 12, "lava": 0.10, "key": 0.25, "steps": 2_000_000}  # Phase 5: Maximum complexity
    ]
    
    log_dir = "./tensorboard_logs/"
    model = None 

    print("START TRAINING: LAVA SMART + MULTIPLE KEYS CURRICULUM")
    
    for i, config in enumerate(stages_config):
        n_rooms = config["rooms"]
        lava_prob = config["lava"]
        key_prob = config["key"]
        steps = config["steps"]
        
        stage_name = f"Lava_Fase{i+1}_{steps}"
        
        print(f"    {stage_name}")
        print(f"   Rooms: {n_rooms}")
        print(f"   Lava Probability: {lava_prob*100}%")
        print(f"   Key Probability: {key_prob*100}%")

        # Create environment
        env = gym.make(
            "MiniGrid-LavaSmartMulti-v0", 
            render_mode=None, 
            n_rooms=n_rooms, 
            lava_prob=lava_prob,
            key_prob=key_prob
        )
        env = ImgObsWrapper(env)

        # Load / Transfer Model
        if model is None:
            if not os.path.exists(initial_model_path):
                print("Error: Pre-trained model not found. Starting training from scratch.")
                return
            
            custom_objects = {"learning_rate": 0.0002, # Low LR to avoid forgetting, moderate entropy for exploration
                            "ent_coef": 0.08,  
                            "clip_range": 0.2}    
            
            model = PPO.load(
                initial_model_path, 
                env=env, 
                custom_objects=custom_objects,
                tensorboard_log=log_dir,
                device= "cpu"
            )
        else:
            print(f"Transferring agent to the next stage")
            model.set_env(env)

        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=500_000,
            save_path=f"./checkpoints/{stage_name}/",
            name_prefix=stage_name
        )

        # Train
        model.learn(
            total_timesteps=steps, 
            callback=checkpoint_callback,
            reset_num_timesteps=True,
            tb_log_name=stage_name
        )

        final_save_name = f"{stage_name}_FINAL"
        model.save(final_save_name)
        print(f"Stage completed. Saved as {final_save_name}.zip")
        env.close()

    print("Training with Multicolor Curriculum Completed")

if __name__ == "__main__":
    train_lava_smart_multi()