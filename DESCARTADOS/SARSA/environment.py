import gymnasium as gym
from gymnasium import spaces
import random
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Door, Key, Lava
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import DIR_TO_VEC, OBJECT_TO_IDX

# --- MAPA DE IDs (Igual que antes) ---
MAPPING = {
    'empty': 0, 'wall': 1, 'agent': 2, 'goal': 3, 
    'key': 4, 'door': 5, 'lava': 7
}

class AutoKey(Key):
    def can_overlap(self):
        return True

class TwoRoomEnv(MiniGridEnv):
    def __init__(self, room_size=7, max_steps=500, agent_view_size=7, **kwargs):
        self.room_size = room_size
        # Ancho para 2 habitaciones: (size-1)*2 + 1
        width = (room_size - 1) * 2 + 1
        height = room_size
        
        mission_space = MissionSpace(mission_func=lambda: "Busca la llave, abre la puerta e ir a la meta")

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=True,
            agent_view_size=agent_view_size,
            **kwargs
        )
        
        self.action_space = spaces.Discrete(4) # Izq, Der, Avanzar, Interactuar (simplificado en step)
        
        # --- OBSERVATION SPACE DE 1 CANAL ---
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(agent_view_size, agent_view_size, 1), 
            dtype=np.uint8
        )
        
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        center_y = height // 2
        
        # --- ESTRUCTURA: 2 HABITACIONES ---
        # Pared vertical separando las dos habitaciones
        split_idx = self.room_size - 1
        self.grid.vert_wall(split_idx, 0)
        
        # 1. PUERTA (Amarilla para destacar)
        # La ponemos en el muro central
        self.grid.set(split_idx, center_y, Door('yellow', is_locked=True))
        
        # 2. LLAVE (Amarilla)
        # La ponemos en la Habitación 1 (Izquierda) en una posición aleatoria
        # Evitamos ponerla justo enfrente de la puerta para que el agente tenga que buscarla
        self.place_obj(
            AutoKey('yellow'), 
            top=(1, 1), 
            size=(self.room_size - 2, self.room_size - 2)
        )

        # 3. AGENTE
        # Empieza en la Habitación 1 (Izquierda)
        self.place_agent(
            top=(1, 1), 
            size=(self.room_size - 2, self.room_size - 2)
        )

        # 4. META
        # En la Habitación 2 (Derecha), al fondo
        self.put_obj(Goal(), width - 2, center_y)

    def get_one_channel_obs(self):
        # ... (Tu lógica original de 1 canal intacta) ...
        obs = self.gen_obs() 
        grid_image = obs['image'] 
        raw_grid = grid_image[:, :, 0]
        
        clean_grid = np.zeros_like(raw_grid)
        clean_grid[raw_grid == OBJECT_TO_IDX['empty']] = MAPPING['empty']
        clean_grid[raw_grid == OBJECT_TO_IDX['wall']] = MAPPING['wall']
        clean_grid[raw_grid == OBJECT_TO_IDX['goal']] = MAPPING['goal']
        clean_grid[raw_grid == OBJECT_TO_IDX['lava']] = MAPPING['lava']
        clean_grid[raw_grid == OBJECT_TO_IDX['door']] = MAPPING['door']
        clean_grid[raw_grid == OBJECT_TO_IDX['key']] = MAPPING['key']
        
        center_x = self.agent_view_size // 2
        bottom_y = self.agent_view_size - 1
        clean_grid[center_x, bottom_y] = MAPPING['agent']
        
        final_obs = np.expand_dims(clean_grid, axis=-1)
        return final_obs.astype(np.uint8)

    def step(self, action):
        action = int(action)
        self.agent_dir = action # Control directo de dirección
        
        dx, dy = DIR_TO_VEC[self.agent_dir]
        front_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
        front_cell = self.grid.get(*front_pos)
        
        reward = -0.1 # Penalización leve por paso
        
        # LÓGICA AUTOMÁTICA (Simplificada para el agente)
        
        # 1. Abrir puerta si tenemos llave y chocamos con ella
        if front_cell and front_cell.type == 'door':
            if front_cell.is_locked and self.carrying:
                front_cell.is_locked = False
                front_cell.is_open = True
                self.carrying = None # Gastamos la llave
                reward += 1.0 # Reward por abrir puerta
                
        # 2. Movimiento
        obs, _, terminated, truncated, info = super().step(2) # 2=MoveForward en Minigrid
        
        # 3. Coger llave automáticamente al pisarla
        curr_cell = self.grid.get(*self.agent_pos)
        if curr_cell and curr_cell.type == 'key':
            self.carrying = curr_cell
            self.grid.set(self.agent_pos[0], self.agent_pos[1], None)
            reward += 1.0 # Reward por encontrar la llave
            
        # 4. Meta
        if terminated and self.grid.get(*self.agent_pos).type == 'goal':
            reward += 10.0 # Gran premio final
            
        return self.get_one_channel_obs(), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self.get_one_channel_obs(), info