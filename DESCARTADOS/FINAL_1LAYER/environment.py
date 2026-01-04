import gymnasium as gym
from gymnasium import spaces
import random
import numpy as np
import pygame
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Door, Key, Lava
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import DIR_TO_VEC, OBJECT_TO_IDX

# --- MAPA DE IDs PERSONALIZADO (1 CANAL) ---
# Esto simplifica lo que ve el agente.
# Minigrid usa sus propios IDs internos, aquí los traducimos a algo simple.
MAPPING = {
    'empty': 0,  # Suelo vacío
    'wall': 1,   # Pared
    'agent': 2,  # El Agente (nosotros)
    'goal': 3,   # Meta (Verde)
    'key': 4,    # Llave
    'door': 5,   # Puerta (Cerrada o Abierta)
    'lava': 7    # Lava (Peligro)
}

class AutoKey(Key):
    def can_overlap(self):
        return True

class SurvivalCorridorEnv(MiniGridEnv):
    def __init__(self, num_rooms=25, room_size=7, max_steps=3000, agent_view_size=7, agent_start_room=0, **kwargs):
        self.num_rooms = num_rooms
        self.room_size = room_size
        width = (room_size - 1) * num_rooms + 1
        height = room_size
        self.agent_start_room = agent_start_room
        
        mission_space = MissionSpace(mission_func=lambda: "Sobrevive y avanza")

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=True,
            agent_view_size=agent_view_size,
            **kwargs
        )
        
        self.action_space = spaces.Discrete(4)
        
        # --- CAMBIO CLAVE: OBSERVATION SPACE DE 1 CANAL ---
        # Ahora la observación es simplemente una matriz de números enteros (0-255)
        # Shape: (Vista, Vista, 1) -> Ejemplo: (7, 7, 1)
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(agent_view_size, agent_view_size, 1), 
            dtype=np.uint8
        )
        
        self.window = None
        self.clock = None

    def set_start_room(self, room_index):
        """Método para cambiar la dificultad desde el Callback"""
        room_id = max(0, min(room_index, self.num_rooms - 1))
        self.agent_start_room = room_id
        self.max_room_reached = room_id 
        return room_id

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        center_y = height // 2
        available_colors = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']

        for i in range(self.num_rooms):
            x_start = i * (self.room_size - 1)
            x_end = x_start + (self.room_size - 1)
            
            # --- DIFICULTAD INVERTIDA ---
            # Room 24 (Inicio del curriculum) = Fácil
            # Room 0 (Meta final) = Difícil
            difficulty_factor = (self.num_rooms - 1 - i) / (self.num_rooms - 1)

            # 1. LAVA
            if i < self.num_rooms - 1: 
                lava_prob = difficulty_factor * 0.9 
                rx, ry = x_start + (self.room_size // 2), center_y
                if random.random() < lava_prob: 
                    self.grid.set(rx, ry, Lava())

            # 2. PUERTAS Y LLAVES
            if i < self.num_rooms - 1:
                self.grid.vert_wall(x_end, 0)
                
                door_prob = difficulty_factor * 0.8 
                
                if random.random() < door_prob:
                    color = available_colors[i % len(available_colors)]
                    self.grid.set(x_end, center_y, Door(color, is_locked=True))
                    
                    # FIX: Llave centrada para no bloquear entrada
                    try: 
                        self.place_obj(
                            AutoKey(color), 
                            top=(x_start + 1, 0),
                            size=(self.room_size - 2, self.room_size)
                        )
                    except: pass
                else: 
                    self.grid.set(x_end, center_y, None) 

        self.agent_pos = (self.agent_start_room * (self.room_size - 1) + 1, center_y)
        self.agent_dir = 0 
        self.put_obj(Goal(), width - 2, center_y)
        
        self.max_x_reached = self.agent_pos[0]
        self.max_room_reached = self.agent_start_room

    def get_one_channel_obs(self):
        """
        Esta es la función mágica.
        Convierte la visión compleja de Minigrid (7x7x3) 
        a una matriz simple de IDs (7x7x1) que tu red entiende fácil.
        """
        # 1. Obtenemos la vista estándar de Minigrid (7, 7, 3)
        # channel 0: object id, channel 1: color, channel 2: state
        obs = self.gen_obs() 
        grid_image = obs['image'] # (7, 7, 3)
        
        # 2. Extraemos solo el canal 0 (IDs de objetos)
        # Minigrid usa estos IDs internos:
        # 1=empty, 2=wall, 8=goal, 9=lava, 4=door, 5=key
        raw_grid = grid_image[:, :, 0] # Ahora es (7, 7)
        
        # 3. (Opcional) Re-mapeamos a tus números favoritos para que sea más limpio
        # Creamos una matriz de ceros
        clean_grid = np.zeros_like(raw_grid)
        
        # Traducimos IDs de Minigrid a los nuestros (definidos arriba en MAPPING)
        clean_grid[raw_grid == OBJECT_TO_IDX['empty']] = MAPPING['empty']
        clean_grid[raw_grid == OBJECT_TO_IDX['wall']] = MAPPING['wall']
        clean_grid[raw_grid == OBJECT_TO_IDX['goal']] = MAPPING['goal']
        clean_grid[raw_grid == OBJECT_TO_IDX['lava']] = MAPPING['lava']
        clean_grid[raw_grid == OBJECT_TO_IDX['door']] = MAPPING['door']
        clean_grid[raw_grid == OBJECT_TO_IDX['key']] = MAPPING['key']
        
        # 4. Ponemos al Agente
        # En la visión egocéntrica de Minigrid, el agente siempre está abajo al centro.
        # coords: (width // 2, height - 1)
        center_x = self.agent_view_size // 2
        bottom_y = self.agent_view_size - 1
        clean_grid[center_x, bottom_y] = MAPPING['agent']
        
        # 5. Añadimos la dimensión del canal para que sea (7, 7, 1) compatible con CNNs
        final_obs = np.expand_dims(clean_grid, axis=-1)
        
        return final_obs.astype(np.uint8)

    def step(self, action):
        # FIX: Asegurar que action es int
        action = int(action)
        
        self.agent_dir = action
        dx, dy = DIR_TO_VEC[self.agent_dir]
        front_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
        front_cell = self.grid.get(*front_pos)
        
        reward = -0.05 
        
        # Abrir puertas
        if front_cell and front_cell.type == 'door' and front_cell.is_locked:
            if self.carrying and self.carrying.color == front_cell.color:
                front_cell.is_locked = False
                front_cell.is_open = True
                self.carrying = None
                reward += 1.0 
        
        # Movimiento base
        obs, _, terminated, truncated, info = super().step(2)
        
        # Coger llaves
        curr_cell = self.grid.get(*self.agent_pos)
        if curr_cell and curr_cell.type == 'key':
            self.carrying = curr_cell
            self.grid.set(self.agent_pos[0], self.agent_pos[1], None)
            reward += 2.0 

        # Recompensa por avanzar
        current_x = self.agent_pos[0]
        if current_x > self.max_x_reached:
            diff = current_x - self.max_x_reached
            reward += diff * 0.1 
            self.max_x_reached = current_x 
            
        current_room = self.agent_pos[0] // (self.room_size - 1)
        if current_room > self.max_room_reached:
            reward += 1.0
            self.max_room_reached = current_room
            
        if terminated and self.grid.get(*self.agent_pos).type == 'goal': 
            reward += 50.0 
            
        if curr_cell and curr_cell.type == 'lava': 
            reward -= 10.0 

        info["max_room"] = self.max_room_reached 
        
        # --- DEVOLVEMOS LA OBSERVACIÓN DE 1 CANAL ---
        return self.get_one_channel_obs(), reward, terminated, truncated, info

    def reset(self, **kwargs):
        # El reset original devuelve un dict, nosotros queremos nuestra matriz
        obs, info = super().reset(**kwargs)
        return self.get_one_channel_obs(), info

    def render(self):
        # Render visual humano (con librería gráfica)
        return super().render()

    def close(self):
        super().close()