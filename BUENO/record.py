import gymnasium as gym
from gymnasium import spaces, ObservationWrapper
from gymnasium.envs.registration import register
from gymnasium.wrappers import RecordVideo  # <--- AÑADIDO: Para grabar
from minigrid.envs.multiroom import MultiRoomEnv
from minigrid.core.world_object import Door, Key
from stable_baselines3 import PPO
import random
import os
from tqdm.auto import tqdm

# =============================================================================
# CONFIGURACIÓN DE USUARIO
# =============================================================================
MODEL_PATH = "checkpoints/3_Nivel_N4_size8/Nivel_2_5_Intermedio_700000_steps.zip"
N_EPISODES = 1000
N_ROOMS = 12
KEY_PROB = 0.0
RENDER_ON_SCREEN = False

# CONFIGURACIÓN DE VIDEO
VIDEO_FOLDER = "videos"
NUM_VIDEOS_TO_RECORD = 10  # Grabaremos las primeras 10 partidas

# =============================================================================
# 1. ENTORNO Y WRAPPERS (Sin cambios significativos)
# =============================================================================
class MulticolorCorridorMax1(MultiRoomEnv):
    def __init__(self, n_rooms=12, key_prob=0.5, **kwargs):
        super().__init__(
            minNumRooms=n_rooms,
            maxNumRooms=n_rooms,
            maxRoomSize=8,
            **kwargs
        )
        self.key_prob = key_prob

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        valid_colors = ['red', 'blue', 'purple', 'yellow', 'grey']
        locked_door_placed = False
        room_indices = list(range(len(self.rooms) - 1))
        random.shuffle(room_indices)

        for i in room_indices:
            room = self.rooms[i]
            if not locked_door_placed and random.random() < self.key_prob:
                door_pos = room.exitDoorPos
                color = random.choice(valid_colors)
                self.grid.set(door_pos[0], door_pos[1], Door(color, is_locked=True))
                self.place_obj(Key(color), top=room.top, size=room.size, max_tries=100)
                locked_door_placed = True

if "MiniGrid-BenchmarkMax1-v0" in gym.envs.registry:
    del gym.envs.registry["MiniGrid-BenchmarkMax1-v0"]

register(
    id="MiniGrid-BenchmarkMax1-v0",
    entry_point=__name__ + ":MulticolorCorridorMax1",
)

class ImgObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        img_space = env.observation_space.spaces["image"]
        self.observation_space = spaces.Box(low=0, high=255, shape=img_space.shape, dtype="uint8")

    def observation(self, obs):
        return obs["image"]

# =============================================================================
# 3. FUNCIÓN DE BENCHMARK CON GRABACIÓN
# =============================================================================
def evaluate_agent():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: No encuentro el archivo del modelo: {MODEL_PATH}")
        return

    # IMPORTANTE: Para grabar video, necesitamos render_mode="rgb_array"
    # Si RENDER_ON_SCREEN es True, usaremos "human", pero RecordVideo no funcionará igual.
    # Por defecto, forzamos rgb_array para que los videos se generen correctamente.
    render_mode = "human" if RENDER_ON_SCREEN else "rgb_array"

    env = gym.make(
        "MiniGrid-BenchmarkMax1-v0",
        render_mode=render_mode,
        n_rooms=N_ROOMS,
        key_prob=KEY_PROB
    )

    # --- AÑADIDO: Wrapper de Grabación ---
    # La función episode_trigger decide qué episodios grabar (aquí: los primeros 10)
    env = RecordVideo(
        env, 
        video_folder=VIDEO_FOLDER,
        episode_trigger=lambda episode_id: episode_id < NUM_VIDEOS_TO_RECORD,
        name_prefix="eval_run"
    )

    env = ImgObsWrapper(env)

    print(f"Cargando modelo: {MODEL_PATH}")
    print(f"Grabando los primeros {NUM_VIDEOS_TO_RECORD} episodios en: /{VIDEO_FOLDER}")

    try:
        model = PPO.load(MODEL_PATH, device='cpu')
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return

    wins = 0
    total_steps_in_wins = []
    pbar = tqdm(range(N_EPISODES), desc="Evaluando")

    for i in pbar:
        obs, _ = env.reset()
        done = False
        steps = 0
        reward_accum = 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            reward_accum = reward # Guardamos el último reward

        if reward_accum > 0:
            wins += 1
            total_steps_in_wins.append(steps)
            result = "Victoria"
        else:
            result = "Derrota"

        current_win_rate = (wins / (i + 1)) * 100
        pbar.set_postfix({"Wins": wins, "Rate": f"{current_win_rate:.1f}%"})

        if (i + 1) % 10 == 0:
            tqdm.write(f"Episodio {i + 1}/{N_EPISODES} | Pasos: {steps} | {result}")

    win_rate = (wins / N_EPISODES) * 100
    avg_steps = sum(total_steps_in_wins) / len(total_steps_in_wins) if total_steps_in_wins else 0

    print("\n" + "=" * 40)
    print(f"TASA DE ÉXITO: {win_rate:.2f}% | Videos guardados en /{VIDEO_FOLDER}")
    
    env.close()

if __name__ == "__main__":
    evaluate_agent()