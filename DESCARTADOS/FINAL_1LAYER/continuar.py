import gymnasium as gym
import torch as th
import torch.nn as nn
import os
import glob
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import RecurrentPPO
from environment import SurvivalCorridorEnv

# --- 1. EXTRACTOR (NECESARIO PARA CARGAR) ---
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

# --- 2. CALLBACK DEL CURRICULUM (LÓGICA MEJORADA) ---
class ReverseCurriculumCallback(BaseCallback):
    def __init__(self, start_room_init=17, win_threshold=0.4, history_len=20, verbose=1):
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

            # UMBRAL_DE_CAMPEON > 0 (Para detectar victorias lejanas)
            is_victory = total_ep_reward > 0.0
            
            self.win_history.append(is_victory)
            
            if is_victory and self.verbose > 0:
                print(f"   🌟 ¡VICTORIA! Room {self.current_start_room} -> Meta | Score: {total_ep_reward:.1f}")

            if len(self.win_history) == self.history_len:
                win_rate = sum(self.win_history) / len(self.win_history)

                # --- AQUI ESTA EL CAMBIO: SI SUPERA EL 0.4 (40%) PASA ---
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
    print("🚑 MODO RESCATE: Cargando checkpoint y bajando dificultad...")

    # --- CONFIGURACIÓN DE RUTAS ---
    checkpoint_dir = "./checkpoints_1canal/"
    
    # BUSCAR EL ÚLTIMO CHECKPOINT AUTOMÁTICAMENTE
    # Busca archivos que empiecen por rl_1ch y terminen en .zip
    list_of_files = glob.glob(os.path.join(checkpoint_dir, 'rl_1ch_*.zip')) 
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"📂 Archivo encontrado más reciente: {latest_file}")
        MODELO_A_CARGAR = latest_file
    else:
        print("❌ No encuentro checkpoints en ./checkpoints_1canal/")
        # Opción B: Cargar el nombre manual si el automático falla
        MODELO_A_CARGAR = "./checkpoints_1canal/rl_1ch_400000_steps.zip" 

    save_path = "./modelos_1canal/"
    os.makedirs(save_path, exist_ok=True)

    # 1. Configurar entorno (Empezamos directamente donde se atascó: 17)
    env = SurvivalCorridorEnv(
        render_mode=None, 
        num_rooms=25, 
        agent_view_size=7, 
        agent_start_room=17 # <--- FORZAMOS INICIO EN LA 17
    )

    # 2. Cargar el modelo
    print(f"🧠 Cargando cerebro desde: {MODELO_A_CARGAR}")
    model = RecurrentPPO.load(MODELO_A_CARGAR, env=env)

    # 3. Callbacks con la nueva configuración
    curriculum_cb = ReverseCurriculumCallback(
        start_room_init=17,   # <--- Empezamos a contar desde la 17
        win_threshold=0.4,    # <--- UMBRAL BAJADO AL 40% (Facilita el avance)
        history_len=20,       # Ventana de 20 partidas
        verbose=1
    )
    
    checkpoint_cb = CheckpointCallback(
        save_freq=50000,
        save_path=checkpoint_dir,
        name_prefix='rl_1ch_resumed' # Cambiamos el prefijo para diferenciar
    )

    print("🏁 Agente re-iniciado. Objetivo: Superar Room 17 con > 40% de victorias.")

    # 4. Seguir entrenando
    # reset_num_timesteps=False es importante para que los logs de Tensorboard sigan la línea continua
    model.learn(
        total_timesteps=1_000_000, 
        callback=CallbackList([curriculum_cb, checkpoint_cb]),
        log_interval=1,
        reset_num_timesteps=False 
    )

    model.save("agente_maestro_1canal_v2")
    print("🏆 ¡ENTRENAMIENTO FINALIZADO!")

if __name__ == "__main__":
    main()