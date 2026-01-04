import gymnasium as gym
from gymnasium import spaces, ObservationWrapper
from gymnasium.envs.registration import register
from minigrid.envs.multiroom import MultiRoomEnv
from minigrid.core.world_object import Door, Key
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import random
import os
from tqdm.auto import tqdm  # <--- IMPORTANTE: Importamos tqdm


# =============================================================================
# 0. CALLBACK PARA LA BARRA DE PROGRESO (NUEVO)
# =============================================================================
class ProgressBarCallback(BaseCallback):
    """
    Callback personalizado para mostrar una barra de progreso con tqdm durante el entrenamiento.
    """

    def __init__(self, total_timesteps, description="Entrenando"):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.description = description
        self.pbar = None

    def _on_training_start(self):
        # Inicializar la barra cuando empieza el entrenamiento
        self.pbar = tqdm(total=self.total_timesteps, desc=self.description, dynamic_ncols=True)

    def _on_step(self):
        # Actualizar la barra. Si usas entornos vectorizados, incrementa por n_envs
        self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self):
        # Cerrar la barra al terminar
        if self.pbar:
            self.pbar.close()


# =============================================================================
# 1. DEFINICIÓN DEL ENTORNO MULTICOLOR
# =============================================================================
class CorredorMulticolor(MultiRoomEnv):
    def __init__(self, n_rooms=4, key_prob=0.2, **kwargs):
        super().__init__(
            minNumRooms=n_rooms,
            maxNumRooms=n_rooms,
            maxRoomSize=10,
            **kwargs
        )
        self.key_prob = key_prob

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        valid_colors = ['red', 'blue', 'purple', 'yellow', 'grey']

        for i, room in enumerate(self.rooms):
            if i == len(self.rooms) - 1:
                break

            if random.random() < self.key_prob:
                door_pos = room.exitDoorPos
                color = random.choice(valid_colors)

                self.grid.set(door_pos[0], door_pos[1], Door(color, is_locked=True))

                self.place_obj(
                    Key(color),
                    top=room.top,
                    size=room.size,
                    max_tries=100
                )


if "MiniGrid-CorredorMulticolor-v0" in gym.envs.registry:
    del gym.envs.registry["MiniGrid-CorredorMulticolor-v0"]

register(
    id="MiniGrid-CorredorMulticolor-v0",
    entry_point=__name__ + ":CorredorMulticolor",
)


# =============================================================================
# 2. WRAPPER DE IMAGEN
# =============================================================================
class ImgObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        img_space = env.observation_space.spaces["image"]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=img_space.shape,
            dtype="uint8"
        )

    def observation(self, obs):
        return obs["image"]


# =============================================================================
# 3. CURRICULUM GRADUAL MULTICOLOR CON TENSORBOARD Y BARRA DE PROGRESO
# =============================================================================
def run_multicolor_curriculum():
    initial_model_path = "Fase_4_Color_12Hab_FINAL2.zip"

    stages = [3, 6, 9, 12]
    steps_per_stage = 2_000_000
    log_dir = "./tensorboard_logs/"

    model = None

    print("🚀 INICIANDO CURRICULUM MULTICOLOR: 3 -> 12 HABITACIONES 🚀")
    print(f"   Pasos por fase: {steps_per_stage}")
    print(f"   TensorBoard Logs: {log_dir}")

    for i, n_rooms in enumerate(stages):
        stage_name = f"Fase_{i + 1}_Color_{n_rooms}Hab"

        print(f"\n--------------------------------------------------")
        print(f"🏁 PREPARANDO {stage_name} ({n_rooms} Habitaciones)")
        # Nota: Quitamos los prints intermedios porque la barra de progreso se encarga de mostrar la actividad

        # 1. Crear entorno
        env = gym.make("MiniGrid-CorredorMulticolor-v0", render_mode=None, n_rooms=n_rooms)
        env = ImgObsWrapper(env)

        # 2. Cargar / Transferir Modelo
        if model is None:
            if not os.path.exists(initial_model_path):
                print(
                    f"❌ ERROR: No encuentro '{initial_model_path}'. Asegúrate de tener el archivo o cambiar el nombre.")
                return

            print(f"🧠 Cargando cerebro base: {initial_model_path}")

            custom_objects = {
                "learning_rate": 0.0001,
                "ent_coef": 0.01
            }

            model = PPO.load(
                initial_model_path,
                env=env,
                custom_objects=custom_objects,
                tensorboard_log=log_dir
            )
        else:
            # print(f"🧠 Transfiriendo agente...") # Comentado para limpiar la salida visual
            model.set_env(env)

        # 3. Checkpoints Callback
        checkpoint_callback = CheckpointCallback(
            save_freq=100_000,
            save_path=f"./checkpoints/{stage_name}2/",
            name_prefix=stage_name,
            verbose=0  # Ponemos verbose 0 para que no ensucie la barra de progreso
        )

        # 4. Progress Bar Callback (NUEVO)
        # Creamos la barra específica para esta fase
        progress_callback = ProgressBarCallback(
            total_timesteps=steps_per_stage,
            description=f"🏃 {stage_name}"
        )

        # 5. Entrenar
        # Pasamos una LISTA de callbacks [checkpoint, progress]
        model.learn(
            total_timesteps=steps_per_stage,
            callback=[checkpoint_callback, progress_callback],
            reset_num_timesteps=True,
            tb_log_name=stage_name,
            progress_bar=False  # Desactivamos la barra nativa de SB3 (si existiera) para usar la nuestra personalizada
        )

        # 6. Guardar final
        final_save_name = f"{stage_name}_FINAL3"
        model.save(final_save_name)
        # Usamos tqdm.write para imprimir sin romper la barra visual si quedara algo pendiente
        tqdm.write(f"✅ {stage_name} COMPLETADA. Guardado en {final_save_name}.zip")

        env.close()

    print("\n🏆 ¡MARATÓN MULTICOLOR COMPLETADO!")


if __name__ == "__main__":
    run_multicolor_curriculum()