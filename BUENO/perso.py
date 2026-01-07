import gymnasium as gym
from gymnasium import spaces, ObservationWrapper
from gymnasium.envs.registration import register
from minigrid.envs.multiroom import MultiRoomEnv
from minigrid.core.world_object import Door, Key
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import random
import os
from tqdm.auto import tqdm  

""" 
Script: Curriculum Learning with Multicolor Corridor in MiniGrid
This script implements a Phased Forward Curriculum Learning approach
to combine the Key-Door skill with navigation environments.
"""

# =============================================================================
# 0. PROGRESS BAR CALLBACK FOR TRAINING
# =============================================================================
class ProgressBarCallback(BaseCallback):

    """
   Custom callback to display a progress bar with tqdm during training.
    """

    def __init__(self, total_timesteps, description="Training"):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.description = description
        self.pbar = None

    def _on_training_start(self): # Initialize the progress bar when training starts
        self.pbar = tqdm(total=self.total_timesteps, desc=self.description, dynamic_ncols=True) 

    def _on_step(self):  # Update the progress bar. If using vectorized environments, increment by n_envs
        self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self): # Close the progress bar when training ends
        if self.pbar:
            self.pbar.close()

# =============================================================================
# 1. CUSTOM MULTICOLOR CORRIDOR ENVIRONMENT
# Environment with several rooms and doors where each door has a different color and requires its corresponding key.
# =============================================================================
class CorredorMulticolor(MultiRoomEnv):

    def __init__(self, n_rooms=4, key_prob=0.2, **kwargs): # Initialize with number of rooms and key probability
        super().__init__(
            minNumRooms=n_rooms, # Minimum number of rooms
            maxNumRooms=n_rooms, # Maximum number of rooms
            maxRoomSize=8, # Maximum room size, this dimensions consider the walls, so the real dimensions of the room are (maxRoomSize-2) x (maxRoomSize-2)
            **kwargs
        )
        self.key_prob = key_prob

    def _gen_grid(self, width, height): # Generate the grid with doors and keys
        super()._gen_grid(width, height)

        valid_colors = ['red', 'blue', 'purple', 'yellow', 'grey']  # List of valid colors for doors and keys

        for i, room in enumerate(self.rooms):   # Iterate through rooms to place doors and keys
            if i == len(self.rooms) - 1:
                break

            if random.random() < self.key_prob: # Place a door with a certain probability
                door_pos = room.exitDoorPos
                color = random.choice(valid_colors) # Choose a random color for the door

                self.grid.set(door_pos[0], door_pos[1], Door(color, is_locked=True)) # Place the door

                self.place_obj( # Place the corresponding key in the room (must be accessible)
                    Key(color),
                    top=room.top,
                    size=room.size,
                    max_tries=100
                )


# Register the custom environment
if "MiniGrid-CorredorMulticolor-v0" in gym.envs.registry:
    del gym.envs.registry["MiniGrid-CorredorMulticolor-v0"]

register(
    id="MiniGrid-CorredorMulticolor-v0",
    entry_point=__name__ + ":CorredorMulticolor",
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
# 3. GRADUAL MULTICOLOR CURRICULUM FUNCTION
# =============================================================================
def run_multicolor_curriculum():
    initial_model_path = "Fase_4_Color_12Hab_FINAL3.zip" # Pre-trained model path

    # Definition of curriculum stages with increasing number of rooms
    stages = [3, 6, 9, 12]
    steps_per_stage = 5_000_000
    log_dir = "./tensorboard_logs/"

    model = None

    for i, n_rooms in enumerate(stages):
        stage_name = f"Fase_{i + 1}_Color_{n_rooms}Hab"

        # 1. Create environment
        env = gym.make("MiniGrid-CorredorMulticolor-v0", render_mode=None, n_rooms=n_rooms)
        env = ImgObsWrapper(env)

        # 2. Load / Transfer Model
        if model is None:
            if not os.path.exists(initial_model_path):
                print(
                    f" ERROR: '{initial_model_path}' not found. Make sure the file exists or change the name.")
                return

            custom_objects = {
                "learning_rate": 0.0001, # Low learning rate to avoid forgetting previous knowledge
                "ent_coef": 0.01 # Moderate entropy coefficient to balance exploration and exploitation
            }

            model = PPO.load(   # Load the initial pre-trained model
                initial_model_path,
                env=env,
                custom_objects=custom_objects,
                tensorboard_log=log_dir
            )
        else:
            model.set_env(env)

        # 3. Checkpoints Callback
        checkpoint_callback = CheckpointCallback(
            save_freq=100_000,  # Save every 100k steps
            save_path=f"./checkpoints/{stage_name}4/",
            name_prefix=stage_name,
            verbose=0  # Set verbose to 0 to avoid cluttering the progress bar
        )

        # 4. Progress Bar Callback (NEW)
        # Create the specific progress bar for this stage
        progress_callback = ProgressBarCallback(
            total_timesteps=steps_per_stage,
            description=f"{stage_name}"
        )

        # 5. Train 
        model.learn(
            total_timesteps=steps_per_stage,
            callback=[checkpoint_callback, progress_callback],
            reset_num_timesteps=True,
            tb_log_name=stage_name,
            progress_bar=False  # Disable native SB3 progress bar (if any) to use our custom one
        )

        # 6. Save final model
        final_save_name = f"{stage_name}_FINAL4"
        model.save(final_save_name)
        tqdm.write(f" {stage_name} COMPLETED. Saved as {final_save_name}.zip")

        env.close()

    print("Training with Multicolor Curriculum Completed")

if __name__ == "__main__":
    run_multicolor_curriculum()