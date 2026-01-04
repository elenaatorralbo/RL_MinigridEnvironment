import gymnasium as gym
from gymnasium import spaces
import random
import numpy as np
import pygame
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Door, Key, Lava
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import DIR_TO_VEC 

class AutoKey(Key):
    def can_overlap(self):
        return True

class SurvivalCorridorEnv(MiniGridEnv):
    def __init__(self, num_rooms=25, room_size=7, max_steps=3000, agent_view_size=9, agent_start_room=0, **kwargs):
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
        self.window = None
        self.clock = None

    def set_start_room(self, room_index):
        self.agent_start_room = room_index

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        center_y = height // 2
        available_colors = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']

        for i in range(self.num_rooms):
            x_start = i * (self.room_size - 1)
            x_end = x_start + (self.room_size - 1)
            
            # --- NUEVA LÓGICA DE DIFICULTAD INVERTIDA ---
            # i = 0 (Inicio real) -> Dificultad Máxima
            # i = 24 (Meta) -> Dificultad Mínima
            
            # Calculamos un factor de dificultad de 1.0 a 0.0
            # Si num_rooms es 25:
            # En room 0: factor = 24/24 = 1.0 (100% difícil)
            # En room 12: factor = 12/24 = 0.5 (50% difícil)
            # En room 24: factor = 0/24 = 0.0 (0% difícil)
            difficulty_factor = (self.num_rooms - 1 - i) / (self.num_rooms - 1)

            # 1. LAVA: Más frecuente cerca del inicio (Room 0), inexistente cerca de la meta
            # Ajustamos para que la meta esté limpia
            if i < self.num_rooms - 1: # Nunca ponemos lava en la ultimísima habitación para no bloquear la meta
                # Probabilidad base de lava escalada por dificultad
                lava_prob = difficulty_factor * 0.9 # Máximo 90% de prob en la Room 0
                
                rx, ry = x_start + (self.room_size // 2), center_y
                if random.random() < lava_prob: 
                    self.grid.set(rx, ry, Lava())

            # 2. PUERTAS Y PAREDES
            if i < self.num_rooms - 1:
                self.grid.vert_wall(x_end, 0)
                
                # Probabilidad de que haya puerta cerrada vs hueco libre
                # Room 0: Alta prob de puerta cerrada (necesita llave/abrir)
                # Room 24: Alta prob de hueco libre (paseo directo)
                door_prob = difficulty_factor * 0.8 # Máximo 80% prob
                
                if random.random() < door_prob:
                    color = available_colors[i % len(available_colors)]
                    self.grid.set(x_end, center_y, Door(color, is_locked=True))
                    try: 
                        self.place_obj(AutoKey(color), top=(x_start + 1, 0), size=(self.room_size - 2, self.room_size))
                    except: pass
                else: 
                    self.grid.set(x_end, center_y, None) # Hueco libre

        self.agent_pos = (self.agent_start_room * (self.room_size - 1) + 1, center_y)
        self.agent_dir = 0 
        self.put_obj(Goal(), width - 2, center_y)
        
        # Reiniciamos métricas
        self.max_x_reached = self.agent_pos[0]
        self.max_room_reached = self.agent_start_room

    def step(self, action):
        self.agent_dir = action
        dx, dy = DIR_TO_VEC[self.agent_dir]
        front_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
        front_cell = self.grid.get(*front_pos)
        
        # 1. Penalización base (Action Penalty)
        reward = -0.05 
        
        # --- LÓGICA DE PUERTAS ---
        if front_cell and front_cell.type == 'door' and front_cell.is_locked:
            if self.carrying and self.carrying.color == front_cell.color:
                front_cell.is_locked = False
                front_cell.is_open = True
                self.carrying = None
                reward += 1.0 
        
        # Ejecutamos movimiento físico
        obs, _, terminated, truncated, info = super().step(2)
        
        # --- LÓGICA DE LLAVES ---
        curr_cell = self.grid.get(*self.agent_pos)
        if curr_cell and curr_cell.type == 'key':
            self.carrying = curr_cell
            self.grid.set(self.agent_pos[0], self.agent_pos[1], None)
            reward += 2.0 

        # --- LÓGICA DE AVANCE (Solo terreno nuevo) ---
        current_x = self.agent_pos[0]
        
        # Si choca con pared: current_x NO cambia, por tanto NO es > max_x_reached -> NO entra -> Reward 0
        # Si retrocede: current_x es menor -> NO entra -> Reward 0
        # Si avanza a lo desconocido: current_x es mayor -> ENTRA -> Reward positivo
        if current_x > self.max_x_reached:
            diff = current_x - self.max_x_reached
            reward += diff * 0.1 
            self.max_x_reached = current_x # Actualizamos el récord
            
        current_room = self.agent_pos[0] // (self.room_size - 1)
        if current_room > self.max_room_reached:
            reward += 1.0
            self.max_room_reached = current_room
            
        if terminated and self.grid.get(*self.agent_pos).type == 'goal': 
            reward += 50.0 
            
        if curr_cell and curr_cell.type == 'lava': 
            reward -= 10.0 

        info["max_room"] = self.max_room_reached 
        return obs, reward, terminated, truncated, info

    def render(self):
        full_img = self.grid.render(tile_size=32, agent_pos=self.agent_pos, agent_dir=self.agent_dir, highlight_mask=None)
        
        view_width_cells = 15
        tile_size = 32
        
        total_width_px = full_img.shape[1]
        view_width_px = view_width_cells * tile_size
        
        agent_x_px = self.agent_pos[0] * tile_size
        
        start_x = agent_x_px - (view_width_px // 2)
        end_x = start_x + view_width_px
        
        if start_x < 0:
            start_x = 0
            end_x = view_width_px
        if end_x > total_width_px:
            end_x = total_width_px
            start_x = total_width_px - view_width_px

        cropped_img = full_img[:, start_x:end_x, :]

        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (cropped_img.shape[1], cropped_img.shape[0])
                )
                pygame.display.set_caption("Survival Pacman")
                self.clock = pygame.time.Clock()

            surf = pygame.surfarray.make_surface(np.transpose(cropped_img, (1, 0, 2)))
            self.window.blit(surf, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(15)

        return cropped_img

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
        super().close()

    # Añade esto dentro de la clase, al final, junto a step, reset, etc.
    def set_start_room(self, room_id):
        """Método para cambiar la dificultad desde el Callback"""
        # Nos aseguramos de que esté entre 0 y 24
        room_id = max(0, min(room_id, self.num_rooms - 1))
        self.agent_start_room = room_id
        # Reiniciamos el récord de migas de pan para la nueva dificultad
        self.max_room_reached = room_id 
        return room_id