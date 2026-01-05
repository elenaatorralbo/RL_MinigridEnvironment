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


# =============================================================================
# 0. CALLBACK (Igual que antes)
# =============================================================================
class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, description="Entrenando"):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.description = description
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc=self.description, dynamic_ncols=True)

    def _on_step(self):
        self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()


# =============================================================================
# 1. ENTORNO (Igual que antes - Necesario para cargar el modelo)
# =============================================================================
class CorredorMulticolor(MultiRoomEnv):
    def __init__(self, n_rooms=4, key_prob=0.2, **kwargs):
        super().__init__(minNumRooms=n_rooms, maxNumRooms=n_rooms, maxRoomSize=10, **kwargs)
        self.key_prob = key_prob

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        valid_colors = ['red', 'blue', 'purple', 'yellow', 'grey']
        for i, room in enumerate(self.rooms):
            if i == len(self.rooms) - 1: break
            if random.random() < self.key_prob:
                door_pos = room.exitDoorPos
                color = random.choice(valid_colors)
                self.grid.set(door_pos[0], door_pos[1], Door(color, is_locked=True))
                self.place_obj(Key(color), top=room.top, size=room.size, max_tries=100)


if "MiniGrid-CorredorMulticolor-v0" in gym.envs.registry:
    del gym.envs.registry["MiniGrid-CorredorMulticolor-v0"]

register(id="MiniGrid-CorredorMulticolor-v0", entry_point=__name__ + ":CorredorMulticolor")


# =============================================================================
# 2. WRAPPER (Igual que antes)
# =============================================================================
class ImgObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        img_space = env.observation_space.spaces["image"]
        self.observation_space = spaces.Box(low=0, high=255, shape=img_space.shape, dtype="uint8")

    def observation(self, obs):
        return obs["image"]


# =============================================================================
# 3. LÓGICA DE REANUDACIÓN
# =============================================================================
def resume_curriculum():
    # --- CONFIGURACIÓN DE REANUDACIÓN ---
    # El modelo que resultó de terminar la Fase 2 (6 Habitaciones)
    model_to_resume_path = "Fase_2_Color_6Hab_FINAL4.zip"

    # Las fases que faltan: 9 habitaciones y 12 habitaciones
    stages_remaining = [9, 12]

    # Mapeo manual para que los nombres de los archivos sigan siendo Fase 3 y Fase 4
    # Aunque sea el elemento 0 y 1 de esta nueva lista
    stage_numbers = [3, 4]

    steps_per_stage = 5_000_000
    log_dir = "./tensorboard_logs/"

    model = None

    print(f"🚀 REANUDANDO CURRICULUM: {stages_remaining} HABITACIONES 🚀")
    print(f"   Cargando desde: {model_to_resume_path}")

    # Iteramos usando zip para tener el número de habitaciones y el número de fase correcto
    for n_rooms, stage_num in zip(stages_remaining, stage_numbers):

        stage_name = f"Fase_{stage_num}_Color_{n_rooms}Hab"

        print(f"\n--------------------------------------------------")
        print(f"🏁 PREPARANDO {stage_name} ({n_rooms} Habitaciones)")

        # 1. Crear entorno con el nuevo número de habitaciones
        env = gym.make("MiniGrid-CorredorMulticolor-v0", render_mode=None, n_rooms=n_rooms)
        env = ImgObsWrapper(env)

        # 2. Cargar modelo (solo la primera vez) o transferir
        if model is None:
            if not os.path.exists(model_to_resume_path):
                print(f"❌ ERROR CRÍTICO: No encuentro '{model_to_resume_path}'.")
                print("   Revisa si el nombre es correcto o busca en la carpeta 'checkpoints' un archivo .zip")
                return

            print(f"🧠 Cargando cerebro de Fase 2: {model_to_resume_path}")

            # Mantenemos los custom objects por si acaso quieres tocar LR,
            # aunque al cargar suele respetar lo guardado salvo que se fuerce.
            custom_objects = {
                "learning_rate": 0.0001,
                "ent_coef": 0.01
            }

            model = PPO.load(
                model_to_resume_path,
                env=env,
                custom_objects=custom_objects,
                tensorboard_log=log_dir
            )
        else:
            print(f"🧠 Transfiriendo agente de la fase anterior...")
            model.set_env(env)

        # 3. Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=100_000,
            save_path=f"./checkpoints/{stage_name}4/",
            name_prefix=stage_name,
            verbose=0
        )

        progress_callback = ProgressBarCallback(
            total_timesteps=steps_per_stage,
            description=f"🏃 {stage_name}"
        )

        # 4. Entrenar
        model.learn(
            total_timesteps=steps_per_stage,
            callback=[checkpoint_callback, progress_callback],
            reset_num_timesteps=True,  # Reseteamos el contador para que empiece de 0 en Tensorboard para esta fase
            tb_log_name=stage_name,
            progress_bar=False
        )

        # 5. Guardar final
        final_save_name = f"{stage_name}_FINAL4"
        model.save(final_save_name)
        tqdm.write(f"✅ {stage_name} COMPLETADA. Guardado en {final_save_name}.zip")

        env.close()

    print("\n🏆 ¡CURRICULUM FINALIZADO CON ÉXITO!")


if __name__ == "__main__":
    resume_curriculum()