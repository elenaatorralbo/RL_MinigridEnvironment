import gymnasium as gym
import minigrid
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium import spaces, ObservationWrapper
import os

""" Script: Training Script for Key-Door Skill in MiniGrid
This script implements the second stage of the Curriculum Learning pipeline.
It leverages a pre-trained Navigation Expert (from the previous stage) and fine-tunes it
in the 'MiniGrid-DoorKey-8x8-v0' environment."""

# =============================================================================
# 1. IMAGE OBSERVATION WRAPPER: to use only preprocessed image observations. This wrapper extracts only 
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
# 2. TRAINING FUNCTION FOR KEY-DOOR ENVIRONMENT
# =============================================================================
def train_doorkey():
    
    env_id = "MiniGrid-DoorKey-8x8-v0" # Environment ID for Key-Door task
    
    prev_model_path = os.path.join("checkpoints", "Nivel_2_5_Intermedio", "Nivel_2_5_Intermedio_500000_steps.zip") #Pre-trained Navigation model
    new_model_name = "KeyDoor"

    log_dir = "./tensorboard_logs_keys/" # Directory for TensorBoard logs

    # 1. Create environment
    env = gym.make(env_id, render_mode=None)
    env = ImgObsWrapper(env)

    # 2. Load pre-trained model
    if os.path.exists(prev_model_path):
        print(f"🧠 Cargando cerebro experto en navegación: {prev_model_path}")
        
        custom_objects = {
            "learning_rate": 0.0001, # Low learning rate to don't forget previous knowledge
            "ent_coef": 0.1 # Higher entropy coefficient to encourage exploration and learning the new skill
        }
        
        model = PPO.load( # Load the previous model
            prev_model_path, 
            env=env, 
            custom_objects=custom_objects,
            tensorboard_log=log_dir 
        )
    else:
        print("Error: Pre-trained model not found. Starting training from scratch.")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    # 3. Save checkpoints during training
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, # Save every 50k steps
        save_path=f"./checkpoints/{new_model_name}/", # Directory to save checkpoints
        name_prefix=new_model_name
    )

    # 4. Train
    model.learn(
        total_timesteps=500_000, # Train for 500k steps
        callback=checkpoint_callback,
        reset_num_timesteps=True,
        tb_log_name="KeyDoor"
    )
    
    model.save(new_model_name)

if __name__ == "__main__":
    train_doorkey()