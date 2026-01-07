import gymnasium as gym
from gymnasium import spaces, ObservationWrapper
from gymnasium.envs.registration import register
from minigrid.envs.multiroom import MultiRoomEnv
from minigrid.core.world_object import Door, Key, Lava, Wall, Goal
from minigrid.core.grid import Grid
from stable_baselines3 import PPO
import random
import os
import time
import collections
from tqdm.auto import tqdm

"""
Script: Evaluation Script for Lava-Enhanced MultiRoom Environment in MiniGrid
This script evaluates a pre-trained PPO agent in a custom MiniGrid environment that 
incorporates lava obstacles along with keys and doors.
Scripted parameters allow customization of the number of rooms, key probability, and lava probability.
Also includes detailed logging of performance metrics and optional on-screen rendering.
"""

# =============================================================================
# PARAMETERS
# =============================================================================
MODEL_PATH = "checkpoints/Lava_Fase1_1000000/Lava_Fase1_1000000_500000_steps.zip"   #Model path to evaluate
N_EPISODES = 1000   # Number of evaluation episodes
N_ROOMS = 6 # Number of rooms
KEY_PROB = 0.1 # Key probability
LAVA_PROB = 0.05  # Lava probability
RENDER_ON_SCREEN = True # Whether to render the environment on screen

# =============================================================================
# 1. ENVIRONMENT (adaptable number of rooms, key probability, and lava probability)
# =============================================================================
class CorredorLavaSmart(MultiRoomEnv):
    def __init__(self, n_rooms=4, lava_prob=0.1, key_prob=0.1, **kwargs):
        super().__init__(
            minNumRooms=n_rooms, 
            maxNumRooms=n_rooms, 
            maxRoomSize=8, 
            **kwargs
        )
        self.key_prob = key_prob
        self.lava_prob = lava_prob

    def _gen_grid(self, width, height):
        max_retries = 500
        for _ in range(max_retries):
            self.grid = Grid(width, height)
            try:
                super()._gen_grid(width, height)
            except Exception:
                continue

            self._place_doors_probabilistic()
            self._add_lava_smart()

            if self._is_path_clear():
                return 

        print("Could not generate a solvable level. Generating without lava.")
        self.lava_prob = 0.0
        super()._gen_grid(width, height)

    def _place_doors_probabilistic(self):
        valid_colors = ['red', 'blue', 'purple', 'yellow', 'grey']
        for i, room in enumerate(self.rooms):
            if i == len(self.rooms) - 1:
                break
            if random.random() < self.key_prob:
                door_pos = room.exitDoorPos
                color = random.choice(valid_colors)
                self.grid.set(door_pos[0], door_pos[1], Door(color, is_locked=True))
                self.place_obj(Key(color), top=room.top, size=room.size, max_tries=100)

    def _add_lava_smart(self):
        safe_cells = set()
        safe_cells.add(self.agent_pos)

        for x in range(self.grid.width):
            for y in range(self.grid.height):
                obj = self.grid.get(x, y)
                if isinstance(obj, (Key, Door, Goal)):
                    safe_cells.add((x, y))
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        safe_cells.add((x + dx, y + dy))

        for room in self.rooms:
            for d_pos in [room.entryDoorPos, room.exitDoorPos]:
                if d_pos:
                    safe_cells.add(d_pos)
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        safe_cells.add((d_pos[0]+dx, d_pos[1]+dy))

            for x in range(room.top[0], room.top[0] + room.size[0]):
                for y in range(room.top[1], room.top[1] + room.size[1]):
                    cell = self.grid.get(x, y)
                    if cell is None and (x, y) not in safe_cells:
                        if random.random() < self.lava_prob:
                            self.grid.set(x, y, Lava())

    def _is_path_clear(self):
        start = self.agent_pos
        end = None
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if isinstance(self.grid.get(x, y), Goal):
                    end = (x, y)
                    break
        if not end: return False 

        queue = collections.deque([start])
        visited = {start}

        while queue:
            x, y = queue.popleft()
            if (x, y) == end: return True
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                    if (nx, ny) not in visited:
                        cell = self.grid.get(nx, ny)
                        if not isinstance(cell, (Lava, Wall)):
                            visited.add((nx, ny))
                            queue.append((nx, ny))
        return False

# Register the custom environment
if "MiniGrid-LavaSmartBenchmark-v0" in gym.envs.registry:
    del gym.envs.registry["MiniGrid-LavaSmartBenchmark-v0"]

register(
    id="MiniGrid-LavaSmartBenchmark-v0",
    entry_point=__name__ + ":CorredorLavaSmart",
)

# =============================================================================
# 2. IMAGE OBSERVATION WRAPPER
# =============================================================================
class ImgObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        img_space = env.observation_space.spaces["image"]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=img_space.shape, dtype="uint8"
        )

    def observation(self, obs):
        return obs["image"]

# =============================================================================
# 3. EVALUATION FUNCTION
# =============================================================================
def evaluate_agent():
    if not os.path.exists(MODEL_PATH):
        return

    render_mode = "human" if RENDER_ON_SCREEN else None

    env = gym.make(
        "MiniGrid-LavaSmartBenchmark-v0",
        render_mode=render_mode,
        n_rooms=N_ROOMS,
        key_prob=KEY_PROB,
        lava_prob=LAVA_PROB
    )
    env = ImgObsWrapper(env)

    try:
        model = PPO.load(MODEL_PATH, device='cpu')
    except Exception as e:
        return

    wins = 0
    total_steps_in_wins = []
    pbar = tqdm(range(N_EPISODES), desc="Evaluating Agent")

    for i in pbar:
        obs, _ = env.reset()
        done = False
        steps = 0
        reward_sum = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            reward_sum = reward

            if RENDER_ON_SCREEN:
                env.render()

        if reward_sum > 0:
            wins += 1
            total_steps_in_wins.append(steps)
            result = "Victory"
        else:
            result = "Defeat/Lava"

        current_win_rate = (wins / (i + 1)) * 100
        pbar.set_postfix({"Wins": wins, "Rate": f"{current_win_rate:.1f}%"})

        if RENDER_ON_SCREEN or (i + 1) % 10 == 0:
            tqdm.write(f"Episode {i+1} | Steps: {steps} | {result}")

    # Final Results
    win_rate = (wins / N_EPISODES) * 100
    avg_steps = sum(total_steps_in_wins) / len(total_steps_in_wins) if total_steps_in_wins else 0

    print("\n" + "="*50)
    print(f"FINAL RESULTS - {MODEL_PATH}")
    print(f"Rooms: {N_ROOMS} | Key Prob.: {KEY_PROB} | Lava Prob.: {LAVA_PROB}")
    print(f"SUCCESS RATE: {win_rate:.2f}% ({wins}/{N_EPISODES})")
    if wins > 0:
        print(f"AVERAGE STEPS (Victories): {avg_steps:.1f}")
    print("="*50)

    env.close()

if __name__ == "__main__":
    evaluate_agent()