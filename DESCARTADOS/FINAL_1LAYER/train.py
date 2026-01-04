import gymnasium as gym
import torch as th
import torch.nn as nn
import os
import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import RecurrentPPO
# REMOVED: from minigrid.wrappers import ImgObsWrapper (Not needed anymore)
from environment import SurvivalCorridorEnv

# --- 1. EXTRACTOR (ADAPTED FOR 1 CHANNEL) ---
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # We know it is 1 channel now because of the new environment
        n_input_channels = 1 
        
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate output dimension dynamically
        with th.no_grad():
            sample_obs = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_obs).shape[1]
            
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Normalize: 0-255 -> 0.0-1.0
        return self.linear(self.cnn(observations.float() / 255.0))

# --- 2. CALLBACK DEL CURRICULUM ---
class ReverseCurriculumCallback(BaseCallback):
    def __init__(self, start_room_init=24, win_threshold=0.5, history_len=20, verbose=1):
        super(ReverseCurriculumCallback, self).__init__(verbose)
        self.current_start_room = start_room_init
        self.win_threshold = win_threshold 
        self.history_len = history_len
        
        self.win_history = deque(maxlen=history_len)
        self.episode_rewards = [] 

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        self.episode_rewards.append(reward)

        if self.locals['dones'][0]:
            total_ep_reward = sum(self.episode_rewards)
            self.episode_rewards = [] 

            # CAMBIO CLAVE:
            # Consideramos victoria si llega a la meta.
            # Como la meta da +50 y la lava quita -10, 
            # cualquier puntuación positiva alta (>10) suele significar victoria.
            # Para estar seguros en distancias largas, > 0 es un buen filtro 
            # (significa que la bonificación de meta superó a los costes de pasos).
            is_victory = total_ep_reward > 0
            
            self.win_history.append(is_victory)
            
            if is_victory and self.verbose > 0:
                print(f"   🌟 ¡VICTORIA! Room {self.current_start_room} -> Meta | Score: {total_ep_reward:.1f}")

            if len(self.win_history) == self.history_len:
                win_rate = sum(self.win_history) / len(self.win_history)

                if win_rate >= self.win_threshold and self.current_start_room > 0:
                    self.current_start_room -= 1
                    self.win_history.clear() 
                    
                    self.training_env.env_method("set_start_room", self.current_start_room)
                    
                    if self.verbose > 0:
                        print("\n" + "="*50)
                        print(f"🚀 ¡NIVEL {self.current_start_room + 1} SUPERADO! Tasa éxito: {win_rate*100:.1f}%")
                        print(f"📍 NUEVO INICIO: HABITACIÓN {self.current_start_room}")
                        print("="*50 + "\n")

        return True

def main():
    print("📉 INICIANDO ENTRENAMIENTO 1 CANAL + CURRICULUM")

    save_path = "./modelos_1canal/"
    checkpoint_dir = "./checkpoints_1canal/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Configurar entorno (Room 24 = Fácil con la nueva lógica invertida)
    env = SurvivalCorridorEnv(
        render_mode=None, 
        num_rooms=25, 
        agent_view_size=7, # IMPORTANT: Matches CNN input size
        agent_start_room=24
    )
    # NO ImgObsWrapper here!

    # 2. Configuración Agente PPO
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        lstm_hidden_size=256,
        n_lstm_layers=1,
    )

    model = RecurrentPPO(
        "CnnLstmPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        
        # --- Hyperparameters ---
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,        # Stable value
        target_kl=0.03,     # Safety brake
        clip_range=0.2,
        ent_coef=0.01,      # Lower entropy is okay for 1-channel (less noise)
        gamma=0.99,
        gae_lambda=0.95,
        tensorboard_log="./logs_1canal/"
    )

    # 3. Callbacks
    curriculum_cb = ReverseCurriculumCallback(
        start_room_init=24, 
        win_threshold=0.4, # 40% Win Rate to advance
        verbose=1
    )
    
    checkpoint_cb = CheckpointCallback(
        save_freq=50000,
        save_path=checkpoint_dir,
        name_prefix='rl_1ch'
    )

    print("🏁 Agente listo. Entrenando con visión simplificada (1 Canal).")

    model.learn(
        total_timesteps=5_000_000, 
        callback=CallbackList([curriculum_cb, checkpoint_cb]),
        log_interval=1 
    )

    model.save("agente_maestro_1canal")
    print("🏆 ¡ENTRENAMIENTO FINALIZADO!")

if __name__ == "__main__":
    main()