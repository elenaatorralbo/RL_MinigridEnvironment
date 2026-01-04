import gymnasium as gym
import torch as th
import torch.nn as nn
import os
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from minigrid.wrappers import ImgObsWrapper
from environment import SurvivalCorridorEnv
from sb3_contrib import RecurrentPPO

# --- 1. EXTRACTOR DE CARACTERÍSTICAS (NECESARIO PARA CARGAR EL MODELO) ---
# Tiene que estar definido exactamente igual que cuando se guardó
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

def main():
    print("🔄 CARGANDO AGENTE EXPERTO PARA FASE FINAL...")

    # Rutas
    model_path = "agente_curriculum_final.zip" # El archivo que generó el script anterior
    save_path = "./mejores_modelos_final/"
    logs_dir = "./tensorboard_logs_final/"
    checkpoint_dir = "./checkpoints_final/"
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # -----------------------------------------------------------
    # CONFIGURACIÓN DEL ENTORNO (MODO DIFÍCIL)
    # -----------------------------------------------------------
    # Forzamos start_room=0. Ya no hay fases fáciles.
    print(">>> Configurando entorno en Habitación 0 (Juego Completo)")
    env = SurvivalCorridorEnv(render_mode=None, num_rooms=25, agent_view_size=9, agent_start_room=0)
    env = ImgObsWrapper(env)

    eval_env = SurvivalCorridorEnv(render_mode=None, num_rooms=25, agent_view_size=9, agent_start_room=0)
    eval_env = ImgObsWrapper(eval_env)

    # -----------------------------------------------------------
    # CARGA DEL CEREBRO
    # -----------------------------------------------------------
    if not os.path.exists(model_path):
        print(f"❌ ERROR: No encuentro el archivo {model_path}")
        print("Asegúrate de que el entrenamiento anterior terminó y guardó el archivo.")
        return

    print(f"🧠 Cargando modelo: {model_path}")
    # Cargamos el modelo. Pasamos el env nuevo para que se conecte.
    model = RecurrentPPO.load(model_path, env=env)
    
    # OPCIONAL: Reducir un poco el learning rate.
    # Como el agente ya sabe "casi todo", aprender despacio ayuda a afinar la puntería.
    # model.learning_rate = 0.0001 

    # -----------------------------------------------------------
    # CALLBACKS (Solo checkpoints y evaluación, sin curriculum)
    # -----------------------------------------------------------
    
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
        name_prefix='rl_model_final',
        save_replay_buffer=True,
        save_vecnormalize=True
    )

    combined_callback = CallbackList([eval_cb, checkpoint_cb])

    # -----------------------------------------------------------
    # ENTRENAMIENTO FINAL
    # -----------------------------------------------------------
    print("🚀 Iniciando 1 Millón de pasos extra en MODO EXPERTO...")
    print("Objetivo: Optimizar la ruta para llegar a la meta antes de los 3000 pasos.")
    
    # reset_num_timesteps=False para que en las gráficas siga sumando (2M -> 3M)
    model.learn(
        total_timesteps=2_000_000, 
        callback=combined_callback, 
        reset_num_timesteps=False 
    )

    model.save("agente_maestro_completado")
    print("¡ENTRENAMIENTO TOTAL FINALIZADO! 🏆")

if __name__ == "__main__":
    main()