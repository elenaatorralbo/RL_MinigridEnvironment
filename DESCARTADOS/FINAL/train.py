import gymnasium as gym
import torch as th
import torch.nn as nn
import os
import shutil # Para limpiar carpetas si quisieras
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, CheckpointCallback
from minigrid.wrappers import ImgObsWrapper
from environment import SurvivalCorridorEnv
from sb3_contrib import RecurrentPPO

# --- 1. EXTRACTOR DE CARACTERÍSTICAS ---
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

# --- 2. CALLBACK DE CURRICULUM OPTIMIZADO ---
class SmartCurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(SmartCurriculumCallback, self).__init__(verbose)
        self.phase = 0 

    def _on_step(self) -> bool:
        # FASE 1: TUTORIAL (Habitación 24) - Hasta 200k
        if self.num_timesteps < 200_000:
            if self.phase != 1:
                print("\n" + "="*50)
                print(">>> FASE 1: TUTORIAL (Habitación 24)")
                print("="*50 + "\n")
                self.phase = 1
            self.training_env.env_method("set_start_room", 24)

        # FASE 1.5: TRANSICIÓN (Habitación 20) - De 200k a 800k
        elif 200_000 <= self.num_timesteps < 800_000:
            if self.phase != 1.5:
                print("\n" + "="*50)
                print(">>> FASE 1.5: TRANSICIÓN SUAVE (Habitación 20)")
                print(">>> Alejándonos un poco (5 habitaciones)...")
                print("="*50 + "\n")
                self.phase = 1.5
            self.training_env.env_method("set_start_room", 20)

        # FASE 2: MEDIA (Habitación 10) - De 800k a 1.4M
        elif 800_000 <= self.num_timesteps < 1_400_000:
            if self.phase != 2:
                print("\n" + "="*50)
                print(">>> FASE 2: DIFICULTAD MEDIA (Habitación 10)")
                print(">>> Mitad del camino (15 habitaciones).")
                print("="*50 + "\n")
                self.phase = 2
            self.training_env.env_method("set_start_room", 10)

        # FASE 2.5: SEMIFINAL (Habitación 7) - De 1.4M a 2.0M
        elif 1_400_000 <= self.num_timesteps < 2_000_000:
            if self.phase != 2.5:
                print("\n" + "="*50)
                print(">>> FASE 2.5: SEMIFINAL (Habitación 7)")
                print(">>> Casi el juego completo.")
                print("="*50 + "\n")
                self.phase = 2.5
            self.training_env.env_method("set_start_room", 7)

        # FASE 3: JUEGO COMPLETO (Habitación 0) - Más de 2.0M
        else:
            if self.phase != 3:
                print("\n" + "="*50)
                print(">>> FASE 3: JUEGO COMPLETO (Habitación 0)")
                print(">>> ¡A POR TODAS!")
                print("="*50 + "\n")
                self.phase = 3
            self.training_env.env_method("set_start_room", 0)

        return True

def main():
    print("🧹 INICIANDO ENTRENAMIENTO DESDE CERO ABSOLUTO...")

    save_path = "./mejores_modelos/"
    logs_dir = "./tensorboard_logs/"
    checkpoint_dir = "./checkpoints/"
    
    # Creamos directorios si no existen
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # -----------------------------------------------------------
    # CONFIGURACIÓN DEL ENTORNO
    # -----------------------------------------------------------
    # Empezamos en start_room=24 para la Fase 1
    env = SurvivalCorridorEnv(render_mode=None, num_rooms=25, agent_view_size=9, agent_start_room=24)
    env = ImgObsWrapper(env)

    # Entorno de evaluación (Habitación 0)
    eval_env = SurvivalCorridorEnv(render_mode=None, num_rooms=25, agent_view_size=9, agent_start_room=0)
    eval_env = ImgObsWrapper(eval_env)

    # -----------------------------------------------------------
    # CREACIÓN DEL MODELO (SIEMPRE NUEVO)
    # -----------------------------------------------------------
    print("🧠 Creando un cerebro nuevo...")
    
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
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.04,
        tensorboard_log=logs_dir
    )

    # -----------------------------------------------------------
    # CALLBACKS
    # -----------------------------------------------------------
    curriculum_cb = SmartCurriculumCallback()
    
    eval_cb = EvalCallback(
        eval_env, 
        best_model_save_path=save_path,
        log_path=logs_dir, 
        eval_freq=10000, 
        deterministic=True, 
        render=False
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50000,
        save_path=checkpoint_dir,
        name_prefix='rl_model',
        save_replay_buffer=True,
        save_vecnormalize=True
    )

    combined_callback = CallbackList([curriculum_cb, eval_cb, checkpoint_cb])

    # -----------------------------------------------------------
    # ENTRENAMIENTO
    # -----------------------------------------------------------
    print("🚀 Despegando...")
    
    model.learn(
        total_timesteps=2_000_000, 
        callback=combined_callback, 
        reset_num_timesteps=True # Esto reinicia los contadores de TensorBoard a 0
    )

    model.save("agente_curriculum_final")
    print("¡Entrenamiento finalizado!")

if __name__ == "__main__":
    main()