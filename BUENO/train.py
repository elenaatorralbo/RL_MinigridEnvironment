import gymnasium as gym
import minigrid
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium import spaces, ObservationWrapper
import os

"""
Script: Curriculum Learning Training Pipeline for MiniGrid
This script implements a Phased Forward Curriculum Learning approach 
to train a PPO agent in increasingly complex MiniGrid-MultiRoom environments.
It utilizes Transfer Learning to pass policy weights between stages.
"""

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
# 2. CURRICULUM CONFIGURATION: to define the phases of the curriculum learning process.
# =============================================================================
def run_final_curriculum():
    
    #Definition of the 3 curriculum stages with increasing complexity
    
    stages = [
        # PHASE 1 - to learn basic navigation in small 2-room environments.
        {
            "name": "1_Nivel_N2",   
            "id": "MiniGrid-MultiRoom-N2-S4-v0", # Environment ID
            "steps": 200_000,  # 200k steps for phase 1
            "kwargs": {} 
        },
        
        # PHASE 2 - to increase complexity with more rooms and larger sizes.
        {
            "name": "2_Nivel_N4_Size5",
            "id": "MiniGrid-MultiRoom-N4-S5-v0", # Environment ID
            "steps": 500_000,  # 500k steps for phase 2
            "kwargs": {} 
        },
        
        # PHASE 3 - final phase with maximum complexity (more rooms and larger sizes).
        {
            "name": "3_Nivel_N4_Size8",
            "id": "MiniGrid-MultiRoom-N4-S8-v0", # Environment ID
            "steps": 700_000, # 700k steps for phase 3
            "kwargs": {"minNumRooms": 6, "maxNumRooms": 6, "maxRoomSize": 8} # Custom kwargs for this environment (6 rooms, max size 8)
        }
    ]

    log_dir = "./tensorboard_logs/" # Directory for TensorBoard logs
    
    model = None
    
    for i, stage in enumerate(stages):
        print(f"\n>>> [PHASE {i+1}/{len(stages)}] RUNNING: {stage['name']}")
        print(f"    Environment: {stage['id']}")
        print(f"    Steps: {stage['steps']}")

        # 1. Create the environment
        env = gym.make(stage["id"], render_mode=None, **stage["kwargs"])
        env = ImgObsWrapper(env)

        # 2. Manage the Model
        if model is None:
            model = PPO(
                "MlpPolicy", # Using MlpPolicy for image observations
                env, 
                verbose=1, 
                learning_rate=0.0003, # Low learning rate for stable training, avoid high variance
                ent_coef=0.01,  # Entropy coefficient to encourage exploration (very important to avoid local minima)
                tensorboard_log=log_dir 
            )
        else:
            model.set_env(env)

        # 3. Checkpoints
        checkpoint_callback = CheckpointCallback(
            save_freq=100000, # Save every 100k steps
            save_path=f"./checkpoints/{stage['name']}/", # Directory to save checkpoints
            name_prefix=stage['name'] 
        )

        # 4. Training
        model.learn(
            total_timesteps=stage['steps'], 
            callback=checkpoint_callback,
            reset_num_timesteps=True,
            tb_log_name=stage['name'] 
        )

        # 5. Save the model at the end of the phase
        save_path = f"{stage['name']}_FINAL" 
        model.save(save_path) # Directory to save
        print(f"✅ PHASE {i+1} COMPLETED. Model saved at: {save_path}.zip")
        
        env.close()

    print("Training completed for all phases.")

if __name__ == "__main__":
    run_final_curriculum()