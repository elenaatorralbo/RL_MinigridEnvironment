import gymnasium as gym
from gymnasium import spaces, ObservationWrapper
from gymnasium.envs.registration import register
from minigrid.envs.multiroom import MultiRoomEnv
from minigrid.core.world_object import Door, Key
from stable_baselines3 import PPO
import random
import os
import time
from tqdm.auto import tqdm  # <--- IMPORTANTE: Importamos la barra de progreso

# =============================================================================
# USER CONFIGURATION
# =============================================================================
# Path to your trained model
MODEL_PATH = "Fase_4_Color_12Hab_FINAL3.zip"

# Number of testing episodes
N_EPISODES = 1000

# Number of rooms (Difficulty)
N_ROOMS = 16

# Set to True to watch the agent play (slower)
# Set to False for fast calculation (turbo mode)
RENDER_ON_SCREEN = False  # <--- Recomendado False para probar la barra a alta velocidad


# =============================================================================
# 1. ENVIRONMENT: MULTICOLOR + MAX 1 KEY
# =============================================================================
class MulticolorCorridorMax1(MultiRoomEnv):
    def __init__(self, n_rooms=12, key_prob=0.00, **kwargs):
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

        # Shuffle room indices so the locked door can appear anywhere
        room_indices = list(range(len(self.rooms) - 1))
        random.shuffle(room_indices)

        for i in room_indices:
            room = self.rooms[i]

            # If we haven't placed a door yet, and the dice roll succeeds...
            if not locked_door_placed and random.random() < self.key_prob:
                door_pos = room.exitDoorPos
                color = random.choice(valid_colors)

                # 1. Place locked door
                self.grid.set(door_pos[0], door_pos[1], Door(color, is_locked=True))

                # 2. Place matching key
                self.place_obj(
                    Key(color),
                    top=room.top,
                    size=room.size,
                    max_tries=100
                )

                # Mark as placed so we do not generate more locked doors
                locked_door_placed = True


# Register the environment
if "MiniGrid-BenchmarkMax1-v0" in gym.envs.registry:
    del gym.envs.registry["MiniGrid-BenchmarkMax1-v0"]

register(
    id="MiniGrid-BenchmarkMax1-v0",
    entry_point=__name__ + ":MulticolorCorridorMax1",
)


# =============================================================================
# 2. OBSERVATION WRAPPER
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
# 3. BENCHMARK FUNCTION
# =============================================================================
def evaluate_agent():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Could not find model file: {MODEL_PATH}")
        return

    # Render configuration
    render_mode = "human" if RENDER_ON_SCREEN else None

    env = gym.make("MiniGrid-BenchmarkMax1-v0", render_mode=render_mode, n_rooms=N_ROOMS)
    env = ImgObsWrapper(env)

    print(f"Loading model: {MODEL_PATH}")
    print(f"Starting evaluation of {N_EPISODES} episodes...")
    print(f"Visual Mode: {'ON' if RENDER_ON_SCREEN else 'OFF (Turbo Mode)'}")

    try:
        model = PPO.load(MODEL_PATH, device="cpu")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    wins = 0
    total_steps_in_wins = []

    # --- INICIO DE LA BARRA DE PROGRESO ---
    # Creamos el iterador con tqdm. 'desc' es el título de la barra.
    pbar = tqdm(range(N_EPISODES), desc="Testing Agent", unit="ep")

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

        # Episode Analysis
        if reward > 0:
            wins += 1
            total_steps_in_wins.append(steps)
            last_result = "Victory"
        else:
            last_result = "Defeat"

        # --- ACTUALIZACIÓN VISUAL DE LA BARRA ---
        # Calculamos el % de victorias actual
        current_win_rate = (wins / (i + 1)) * 100

        # Actualizamos la información al lado de la barra (Wins y Win Rate)
        pbar.set_postfix({
            "Wins": wins,
            "Rate": f"{current_win_rate:.1f}%",
            "Last": f"{last_result} ({steps}s)"
        })

        # Si tienes RENDER_ON_SCREEN en True, usamos tqdm.write en lugar de print
        # para no "romper" la barra visualmente.
        if RENDER_ON_SCREEN:
            tqdm.write(f"Ep {i + 1}: {last_result} in {steps} steps")

    # --- FINAL CALCULATIONS ---
    win_rate = (wins / N_EPISODES) * 100
    avg_steps = sum(total_steps_in_wins) / len(total_steps_in_wins) if total_steps_in_wins else 0

    print("\n" + "=" * 40)
    print(f"Rooms: {N_ROOMS}")
    print(f"TOTAL WINS:       {wins} / {N_EPISODES}")
    print(f"SUCCESS RATE:     {win_rate:.2f}%")
    if wins > 0:
        print(f"AVERAGE STEPS:    {avg_steps:.1f} (in winning episodes)")

    env.close()


if __name__ == "__main__":
    evaluate_agent()