import gymnasium as gym
from gymnasium import spaces
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Wall, Floor, Lava, Key, Door, Goal
from minigrid.minigrid_env import MiniGridEnv
import random

class SequentialRoomsEnv(MiniGridEnv):
    def __init__(
        self,
        n_rooms=20,
        room_size=5,  # Tamaño interno (sin paredes)
        max_steps=2000,
        start_room_index=0, # Para el curriculum learning
        render_mode="human"
    ):
        # Tamaño total: (n_rooms * (room_size + 1)) + 1 (pared final)
        # Altura: room_size + 2 (paredes arriba y abajo)
        self.n_rooms = n_rooms
        self.room_internal_size = room_size
        # El ancho de una "unidad de habitación" es room_size + 1 (pared compartida)
        width = (n_rooms * (room_size + 1)) + 1
        height = room_size + 2
        
        self.start_room_index = start_room_index
        self.max_room_reached = 0
        self.has_key_obs = 0 # Estado interno para la observación
        
        # Mapping para la observación simplificada
        self.OBJECT_TO_IDX = {
            "empty": 0, "wall": 1, "lava": 2, "door_closed": 3, 
            "door_open": 4, "key": 5, "goal": 6, "agent": 7
        }

        mission_space = MissionSpace(mission_func=lambda: "Reach the goal")
        
        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=max_steps,
            render_mode=render_mode,
            see_through_walls=False
        )
        
        # Definir espacio de observación personalizado
        # 7x7 visión + 2 (vector dx, dy al objetivo inmediato) + 1 (tiene llave)
        self.observation_space = spaces.Box(
            low=-999, high=999, shape=(7*7 + 3,), dtype=np.int32
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        
        center_y = height // 2 

        # Generar habitaciones
        for i in range(1, self.n_rooms):
            x_wall = i * (self.room_internal_size + 1)
            # Pared vertical
            for y in range(1, height - 1):
                self.grid.set(x_wall, y, Wall())
            
            difficulty = i / self.n_rooms 
            prob_door = 0.2 + (0.6 * difficulty)
            prob_lava = 0.1 + (0.4 * difficulty)

            # 1. Puertas y Llaves
            if self.unwrapped.np_random.random() < prob_door:
                door = Door(color='yellow', is_locked=True)
                self.grid.set(x_wall, center_y, door)
                
                # Rango de la habitación anterior
                prev_room_start_x = (i - 1) * (self.room_internal_size + 1) + 1
                
                # --- CORRECCIÓN CRÍTICA AQUÍ ---
                while True:
                    key_x = self.unwrapped.np_random.integers(prev_room_start_x, x_wall)
                    key_y = self.unwrapped.np_random.integers(1, height - 1)
                    
                    # Verificar que la celda está vacía
                    is_empty = self.grid.get(key_x, key_y) is None
                    
                    # Verificar que NO es la posición de inicio del agente en esa habitación
                    # (El agente siempre empieza en la columna prev_room_start_x y altura center_y)
                    is_agent_start = (key_x == prev_room_start_x) and (key_y == center_y)
                    
                    if is_empty and not is_agent_start:
                        self.grid.set(key_x, key_y, Key('yellow'))
                        break
            else:
                self.grid.set(x_wall, center_y, None)

            # 2. Lava
            num_lava = int(prob_lava * 3) 
            for _ in range(num_lava):
                lx = self.unwrapped.np_random.integers(x_wall + 1, x_wall + self.room_internal_size + 1)
                ly = self.unwrapped.np_random.integers(1, height - 1)
                
                if ly != center_y and self.grid.get(lx, ly) is None:
                    self.grid.set(lx, ly, Lava())

        # Colocar Meta
        self.put_obj(Goal(), width - 2, center_y)

        # Colocar Agente
        start_x_room = self.start_room_index * (self.room_internal_size + 1) + 1
        self.agent_pos = (start_x_room, center_y)
        self.agent_dir = 0
        
        self.max_room_reached = self.start_room_index

    def step(self, action):
        # 1. Guardar estado PREVIO
        old_pos = self.agent_pos
        # Comprobamos si ANTES de movernos ya teníamos la llave
        had_key_before = (self.carrying and isinstance(self.carrying, Key))
        
        # Checkear estado de la puerta ANTES de la acción (para evitar spam de abrir/cerrar)
        # Obtenemos la celda enfrente
        front_cell = self.grid.get(*self.front_pos)
        door_open_before = False
        if isinstance(front_cell, Door) and front_cell.is_open:
            door_open_before = True

        # 2. Ejecutar paso en MiniGrid
        obs, reward, terminated, truncated, info = super().step(action)
        
        # --- Cálculo de Recompensa Personalizada ---
        custom_reward = 0
        
        # 3. Castigo por paso (Living Penalty) para evitar que se quede quieto
        # Si el entorno devolvió 0, le forzamos un negativo pequeño
        if reward == 0:
            custom_reward -= 0.01 
        
        # 4. Movimiento izquierda/derecha
        if self.agent_pos[0] > old_pos[0]:
            custom_reward += 0.5  # Avanzar derecha
        elif self.agent_pos[0] < old_pos[0]:
            custom_reward -= 1.0  # Retroceder izquierda
            
        # 5. Lógica de la LLAVE (CORREGIDA)
        # Comprobamos si AHORA tenemos la llave
        has_key_now = (self.carrying and isinstance(self.carrying, Key))
        
        # Solo damos premio si NO la tenía antes Y la tiene ahora
        if has_key_now and not had_key_before:
             custom_reward += 2.0 
        
        # 6. Lógica de la PUERTA (CORREGIDA)
        # Miramos la celda que tenemos delante ahora (o la que teníamos, la posición no cambia al abrir)
        # Nota: front_pos se actualiza en super().step si nos hemos girado, 
        # pero para abrir puerta no nos giramos.
        current_front_cell = self.grid.get(*self.front_pos)
        door_open_now = False
        if isinstance(current_front_cell, Door) and current_front_cell.is_open:
            door_open_now = True
            
        # Solo premiar si estaba cerrada y ahora está abierta
        if door_open_now and not door_open_before:
             custom_reward += 3.0
        
        # 7. Nueva habitación alcanzada
        current_room = self.agent_pos[0] // (self.room_internal_size + 1)
        if current_room > self.max_room_reached:
            custom_reward += 10.0 
            self.max_room_reached = current_room
            
        # 8. Muerte por Lava o Meta
        if terminated:
            current_cell = self.grid.get(self.agent_pos[0], self.agent_pos[1])
            if isinstance(current_cell, Lava):
                custom_reward -= 50.0 
                reward = 0 
            elif isinstance(current_cell, Goal):
                custom_reward += 100.0
            
        total_reward = reward + custom_reward
        
        # Actualizar observación personalizada
        obs = self.gen_custom_obs()
        
        return obs, total_reward, terminated, truncated, info

    def gen_custom_obs(self):
        """
        Genera la matriz 7x7 simplificada + vector heurístico
        """
        # 1. Obtener visión local 7x7 (MiniGrid standard method)
        grid, vis_mask = self.gen_obs_grid(agent_view_size=7)
        
        # Codificar grid a enteros simples
        simple_grid = np.zeros((7, 7), dtype=np.int32)
        for i in range(7):
            for j in range(7):
                obj = grid.get(i, j)
                val = 0
                if obj is None: val = self.OBJECT_TO_IDX["empty"]
                elif isinstance(obj, Wall): val = self.OBJECT_TO_IDX["wall"]
                elif isinstance(obj, Lava): val = self.OBJECT_TO_IDX["lava"]
                elif isinstance(obj, Door): 
                    val = self.OBJECT_TO_IDX["door_open"] if obj.is_open else self.OBJECT_TO_IDX["door_closed"]
                elif isinstance(obj, Key): val = self.OBJECT_TO_IDX["key"]
                elif isinstance(obj, Goal): val = self.OBJECT_TO_IDX["goal"]
                simple_grid[i, j] = val
        
        # 2. Información vectorial útil (Heurística)
        # Buscar objetivo inmediato:
        # - Si hay puerta cerrada visible -> objetivo es puerta (o llave si no la tengo)
        # - Si tengo llave -> objetivo es puerta
        # - Default -> Meta (Derecha)
        
        target_pos = (self.width - 2, self.height // 2) # Default goal
        
        has_key = 1 if (self.carrying and isinstance(self.carrying, Key)) else 0
        
        # Lógica simplificada de "radar" para ExpectedSARSA
        # Calculamos vector relativo (dx, dy) hacia la meta global
        dx = target_pos[0] - self.agent_pos[0]
        dy = target_pos[1] - self.agent_pos[1]
        
        # Aplanar todo
        obs_vector = np.concatenate([
            simple_grid.flatten(), # 49 valores
            [dx, dy, has_key]      # 3 valores extra
        ])
        
        return obs_vector

    def set_start_room(self, room_idx):
        self.start_room_index = room_idx