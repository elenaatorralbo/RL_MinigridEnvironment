import gymnasium as gym
import torch as th
import torch.nn as nn
import os
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Importamos la nueva clase
from environment import TwoRoomEnv

# --- 1. EXTRACTOR (Igual que el tuyo) ---
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = 1 
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
            sample_obs = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_obs).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations.float() / 255.0))

def main():
    print("🔑 INICIANDO ENTRENAMIENTO: ESCENARIO 2 HABITACIONES")

    save_path = "./modelos_2rooms/"
    os.makedirs(save_path, exist_ok=True)

    # 1. Configurar entorno de 2 habitaciones
    env = TwoRoomEnv(
        render_mode=None, 
        room_size=8,        # Tamaño cómodo para moverse
        agent_view_size=7   # Vista 7x7
    )

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
        learning_rate=0.0005,  # SUBIR un poco (antes 0.0003)
        n_steps=2048,          # SUBIR: Más memoria de lo que ha pasado ayuda a LSTM
        batch_size=64,
        ent_coef=0.05,         # ¡CLAVE! Subir de 0.01 a 0.05 para forzar exploración
        gamma=0.95,            # Un poco impaciente, pero no demasiado
        gae_lambda=0.95,
        tensorboard_log="./logs_2rooms/"
    )
    
    # 3. Guardado periódico
    checkpoint_cb = CheckpointCallback(
        save_freq=20000, 
        save_path=save_path, 
        name_prefix='rl_2room'
    )

    print("🏁 Agente listo. Objetivo: Coger llave -> Abrir puerta -> Meta.")

    # Entrenamos menos pasos porque el problema es más fácil que el corredor infinito
    model.learn(
        total_timesteps=500_000, 
        callback=checkpoint_cb,
        log_interval=4
    )

    model.save("agente_maestro_2rooms")
    print("🏆 ¡ENTRENAMIENTO FINALIZADO!")

if __name__ == "__main__":
    main()