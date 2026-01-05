import gymnasium as gym
from gymnasium import spaces, ObservationWrapper
from gymnasium.envs.registration import register
from minigrid.envs.multiroom import MultiRoomEnv
from minigrid.core.world_object import Door, Key
from stable_baselines3 import PPO
import random
import os
import time
from tqdm.auto import tqdm  # <--- AÑADIDO: Importamos la barra

# =============================================================================
# CONFIGURACIÓN DE USUARIO
# =============================================================================
# Ruta al modelo entrenado
MODEL_PATH = "Fase_4_Color_12Hab_FINAL3.zip"

# Número de episodios de test
N_EPISODES = 1000

# Número de habitaciones (Dificultad)
N_ROOMS = 12

# PROBABILIDAD DE QUE APAREZCA UNA LLAVE (0.0 a 1.0)
KEY_PROB = 0.5

# Ver al agente jugar en pantalla (True) o cálculo rápido (False)
RENDER_ON_SCREEN = True


# =============================================================================
# 1. ENTORNO: MULTICOLOR + MAX 1 LLAVE
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

        # Barajamos las habitaciones para que la puerta cerrada pueda estar en cualquiera
        room_indices = list(range(len(self.rooms) - 1))
        random.shuffle(room_indices)

        for i in room_indices:
            room = self.rooms[i]

            # Si aún no hemos puesto puerta Y el dado de probabilidad acierta...
            if not locked_door_placed and random.random() < self.key_prob:
                door_pos = room.exitDoorPos
                color = random.choice(valid_colors)

                # 1. Ponemos la puerta cerrada
                self.grid.set(door_pos[0], door_pos[1], Door(color, is_locked=True))

                # 2. Ponemos la llave del mismo color
                self.place_obj(
                    Key(color),
                    top=room.top,
                    size=room.size,
                    max_tries=100
                )

                # Marcamos como puesta para no generar más de una
                locked_door_placed = True


# Registro del entorno
if "MiniGrid-BenchmarkMax1-v0" in gym.envs.registry:
    del gym.envs.registry["MiniGrid-BenchmarkMax1-v0"]

register(
    id="MiniGrid-BenchmarkMax1-v0",
    entry_point=__name__ + ":MulticolorCorridorMax1",
)


# =============================================================================
# 2. WRAPPER DE OBSERVACIÓN
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
# 3. FUNCIÓN DE BENCHMARK
# =============================================================================
def evaluate_agent():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: No encuentro el archivo del modelo: {MODEL_PATH}")
        return

    # Configuración de renderizado
    render_mode = "human" if RENDER_ON_SCREEN else None

    # --- AQUÍ PASAMOS LA PROBABILIDAD DE LLAVE ---
    env = gym.make(
        "MiniGrid-BenchmarkMax1-v0",
        render_mode=render_mode,
        n_rooms=N_ROOMS,
        key_prob=KEY_PROB  # <--- IMPORTANTE: Usamos la variable de config
    )
    env = ImgObsWrapper(env)

    print(f"Cargando modelo: {MODEL_PATH}")
    print(f"Iniciando evaluación de {N_EPISODES} episodios...")
    print(f"Probabilidad de llave: {KEY_PROB * 100}%")
    print(f"Modo Visual: {'ON' if RENDER_ON_SCREEN else 'OFF (Modo Turbo)'}")

    try:
        model = PPO.load(MODEL_PATH, device='cpu')
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return

    wins = 0
    total_steps_in_wins = []

    # --- CAMBIO AQUÍ: Creamos la barra de progreso ---
    pbar = tqdm(range(N_EPISODES), desc="Evaluando")

    for i in pbar:
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            if RENDER_ON_SCREEN:
                env.render()

        # Análisis del episodio
        result = "Derrota"
        if reward > 0:
            wins += 1
            result = "Victoria"
            total_steps_in_wins.append(steps)

        # --- CAMBIO AQUÍ: Actualizamos estadísticas en la barra ---
        current_win_rate = (wins / (i + 1)) * 100
        pbar.set_postfix({"Wins": wins, "Rate": f"{current_win_rate:.1f}%"})

        # Imprimir progreso cada 10 episodios o si el modo visual está activo
        if RENDER_ON_SCREEN or (i + 1) % 10 == 0:
            # Usamos tqdm.write para no romper la barra visual
            tqdm.write(f"Episodio {i + 1}/{N_EPISODES} | Pasos: {steps} | {result}")

    # --- CÁLCULOS FINALES ---
    win_rate = (wins / N_EPISODES) * 100
    avg_steps = sum(total_steps_in_wins) / len(total_steps_in_wins) if total_steps_in_wins else 0

    print("\n" + "=" * 40)
    print(f"Habitaciones: {N_ROOMS}")
    print(f"Probabilidad Llave: {KEY_PROB}")
    print(f"VICTORIAS TOTALES:  {wins} / {N_EPISODES}")
    print(f"TASA DE ÉXITO:      {win_rate:.2f}%")
    if wins > 0:
        print(f"PASOS PROMEDIO:     {avg_steps:.1f} (en victorias)")

    env.close()


if __name__ == "__main__":
    evaluate_agent()