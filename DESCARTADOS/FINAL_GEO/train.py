import gymnasium as gym
import torch as th
import torch.nn as nn
import os
import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import RecurrentPPO
from minigrid.wrappers import ImgObsWrapper
from environment import SurvivalCorridorEnv

# --- 1. EXTRACTOR ---
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations.float() / 10.0))

# --- 2. CALLBACK DEL CURRICULUM (MODIFICADO) ---
class ReverseCurriculumCallback(BaseCallback):
    def __init__(self, start_room_init=24, win_threshold=0.35, history_len=20, verbose=1):
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

            # UMBRAL_DE_CAMPEON: Puntos necesarios para considerar que ha ganado
            UMBRAL_DE_CAMPEON = 40.0 
            
            is_victory = total_ep_reward > UMBRAL_DE_CAMPEON
            self.win_history.append(is_victory)
            
            # CHIVATO: Si gana una partida, te avisa
            if is_victory and self.verbose > 0:
                print(f"   🌟 ¡VICTORIA INDIVIDUAL! Score: {total_ep_reward:.1f}")

            if len(self.win_history) == self.history_len:
                win_rate = sum(self.win_history) / len(self.win_history)

                # Si ganamos el 35% de las veces...
                if win_rate >= self.win_threshold and self.current_start_room > 0:
                    self.current_start_room -= 1
                    self.win_history.clear() 
                    
                    self.training_env.env_method("set_start_room", self.current_start_room)
                    
                    if self.verbose > 0:
                        print("\n" + "="*50)
                        print(f"🚀 ¡NIVEL {self.current_start_room + 1} SUPERADO! Tasa éxito: {win_rate*100:.1f}%")
                        print(f"📍 SALIENDO EN LA HABITACIÓN: {self.current_start_room}")
                        print("="*50 + "\n")

        return True

def main():
    print("📉 INICIANDO ENTRENAMIENTO 'INICIO DESLIZANTE' (FIX ANTIASTASCO)")

    save_path = "./modelos_curriculum_inverso/"
    checkpoint_dir = "./checkpoints_inverso/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # -------------------------------------------------------------
    # OPCIONAL: Si quieres seguir donde lo dejaste, cambia esto a 23.
    # Si empiezas de cero, déjalo en 24.
    EMPEZAR_EN_HABITACION = 24 
    # -------------------------------------------------------------

    env = SurvivalCorridorEnv(render_mode=None, num_rooms=25, agent_view_size=9, agent_start_room=EMPEZAR_EN_HABITACION)
    env = ImgObsWrapper(env)
    
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
        learning_rate=0.0003,
        
        # --- CAMBIOS CLAVE AQUÍ ---
        ent_coef=0.03,     # <--- Subido a 0.03 (Más exploración)
        n_steps=2048,      
        batch_size=64,     
        n_epochs=20,       # <--- AÑADIDO: Repasa 20 veces cada lección (aprende más rápido de las victorias)
        # --------------------------
        
        gamma=0.99,        
        gae_lambda=0.95,   
        tensorboard_log="./logs_curriculum_inverso/"
    )

    # Callbacks (Umbral bajado a 0.35)
    curriculum_cb = ReverseCurriculumCallback(
        start_room_init=EMPEZAR_EN_HABITACION, 
        win_threshold=0.5, # <--- CAMBIO CLAVE: Solo necesita ganar el 50% de las veces
        verbose=1
    )
    
    checkpoint_cb = CheckpointCallback(
        save_freq=50000,
        save_path=checkpoint_dir,
        name_prefix='rl_inverso'
    )

    print("🏁 Agente listo. Configuración 'Desatascador' activada.")

    model.learn(
        total_timesteps=2_000_000, 
        callback=CallbackList([curriculum_cb, checkpoint_cb]),
        log_interval=1  
    )

    model.save("agente_maestro_inverso")
    print("🏆 ¡ENTRENAMIENTO FINALIZADO!")

if __name__ == "__main__":
    main()