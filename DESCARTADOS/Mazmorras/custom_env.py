import gymnasium as gym
from gymnasium import spaces
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Key, Door, Wall, Lava
from minigrid.minigrid_env import MiniGridEnv

class CurriculumTempleEnv(MiniGridEnv):
    def __init__(self, render_mode=None, max_steps=1000):
        self.grid_w = 19
        self.grid_h = 19
        self.current_level = 1 
        
        # Historial de recompensas
        self.rewards_history = {
            'got_yellow': False, 'opened_yellow': False, 
            'got_red': False, 'opened_red': False, 
            'got_blue': False, 'opened_blue': False
        }
        
        # Misión fija para evitar errores de texto
        mission_space = MissionSpace(mission_func=lambda: "traverse the temple")
        
        super().__init__(
            mission_space=mission_space,
            width=self.grid_w,
            height=self.grid_h,
            max_steps=max_steps, 
            render_mode=render_mode
        )
        
        # --- CAMBIO IMPORTANTE: 6 ACCIONES ARCADE ---
        # 0:Izq, 1:Der, 2:Arr, 3:Abj, 4:Coger, 5:Abrir
        self.action_space = spaces.Discrete(6)

    def set_level(self, level):
        if self.current_level != level:
            print(f"--- Environment switching to Level {level} ---")
            self.current_level = level

    def reset(self, *, seed=None, options=None):
        # Reiniciar historial
        self.rewards_history = {
            'got_yellow': False, 'opened_yellow': False, 
            'got_red': False, 'opened_red': False, 
            'got_blue': False, 'opened_blue': False
        }
        
        obs, info = super().reset(seed=seed, options=options)
        
        # Inicializar distancia previa para reward shaping
        target_pos = self._get_target_pos()
        self.prev_dist = np.abs(np.array(self.agent_pos) - np.array(target_pos)).sum()
        
        return obs, info

    def _gen_grid(self, width, height):
        # --- TU MAPA ORIGINAL EXACTO ---
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # PAREDES
        self.grid.vert_wall(6, 0); self.grid.vert_wall(12, 0)
        self.grid.horz_wall(0, 6); self.grid.horz_wall(0, 12)

        # HUECOS
        self.grid.set(6, 3, None); self.grid.set(12, 3, None) 
        self.grid.set(12, 9, None); self.grid.set(6, 9, None) 
        self.grid.set(6, 15, None) 

        # PUERTAS
        self.door_yellow = Door('yellow', is_locked=True); self.grid.set(15, 6, self.door_yellow) 
        self.door_red = Door('red', is_locked=True); self.grid.set(3, 12, self.door_red)
        self.door_blue = Door('blue', is_locked=True); self.grid.set(12, 15, self.door_blue)

        # LLAVES
        self.key_yellow = Key('yellow'); self.place_obj(self.key_yellow, top=(7, 1), size=(5, 5))
        self.key_red = Key('red'); self.place_obj(self.key_red, top=(7, 7), size=(5, 5))
        self.key_blue = Key('blue'); self.place_obj(self.key_blue, top=(7, 13), size=(5, 5))

        # META
        self.place_obj(Goal(), top=(13, 13), size=(5, 5))
        self.goal_pos = np.array([15, 15]) 

        # LAVA
        lava_density = 0.3
        # Solo generamos lava en niveles altos para no frustrar la grabación inicial
        # (O puedes dejarlo como estaba si eres muy pro jugando)
        if self.current_level >= 1: 
            for col_room in range(3):
                for row_room in range(3):
                    if col_room == 2 and row_room == 2: continue 
                    base_x = col_room * 6
                    base_y = row_room * 6
                    for x in range(base_x + 2, base_x + 5):
                        for y in range(base_y + 2, base_y + 5):
                            if self.grid.get(x, y) is None:
                                if self.np_random.uniform(0, 1) < lava_density:
                                    self.grid.set(x, y, Lava())

        # SPAWN
        if self.current_level == 1: self.place_agent(top=(13, 13), size=(5, 5))
        elif self.current_level == 2: self.place_agent(top=(1, 13), size=(5, 5))
        elif self.current_level == 3: self.place_agent(top=(13, 7), size=(5, 5))
        else: self.place_agent(top=(1, 1), size=(5, 5))
        
        self.mission = "traverse the temple"

    def _get_target_pos(self):
        """ TU LÓGICA MÁGICA DE DISTANCIAS """
        def safe_pos(obj):
            if obj.cur_pos is not None: return obj.cur_pos
            return self.agent_pos

        if self.current_level == 1: return self.goal_pos

        if self.current_level == 2:
            if not self.rewards_history['got_blue']: return safe_pos(self.key_blue)
            elif not self.door_blue.is_open: return safe_pos(self.door_blue)
            else: return self.goal_pos

        if self.current_level == 3:
            if not self.rewards_history['got_red']: return safe_pos(self.key_red)
            elif not self.door_red.is_open: return safe_pos(self.door_red)
            elif not self.rewards_history['got_blue']: return safe_pos(self.key_blue)
            elif not self.door_blue.is_open: return safe_pos(self.door_blue)
            else: return self.goal_pos

        if self.current_level == 4:
            if not self.rewards_history['got_yellow']: return safe_pos(self.key_yellow)
            elif not self.door_yellow.is_open: return safe_pos(self.door_yellow)
            elif not self.rewards_history['got_red']: return safe_pos(self.key_red)
            elif not self.door_red.is_open: return safe_pos(self.door_red)
            elif not self.rewards_history['got_blue']: return safe_pos(self.key_blue)
            elif not self.door_blue.is_open: return safe_pos(self.door_blue)
            else: return self.goal_pos
        
        return self.goal_pos

    def step(self, action):
        # --- AQUÍ ESTÁ LA MAGIA ARCADE INTEGRADA EN TU MAPA ---
        reward = 0
        terminated = False
        truncated = False
        
        # 1. Calcular objetivo ANTES de movernos (para tu reward shaping)
        old_target_pos = self._get_target_pos()

        # 2. MOVIMIENTO ARCADE DIRECTO
        target_pos = None
        if action == 0: # IZQUIERDA
            self.agent_dir = 2
            target_pos = self.agent_pos + np.array([-1, 0])
        elif action == 1: # DERECHA
            self.agent_dir = 0
            target_pos = self.agent_pos + np.array([1, 0])
        elif action == 2: # ARRIBA
            self.agent_dir = 3
            target_pos = self.agent_pos + np.array([0, -1])
        elif action == 3: # ABAJO
            self.agent_dir = 1
            target_pos = self.agent_pos + np.array([0, 1])

        # Ejecutar Movimiento
        if action < 4 and target_pos is not None:
            cell = self.grid.get(*target_pos)
            if cell is None or cell.can_overlap():
                self.agent_pos = target_pos
            if cell is not None and cell.type == 'lava':
                terminated = True
                reward = -10

        # 3. ACCIÓN COGER (PICKUP)
        elif action == 4:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
                    
                    # Tus rewards personalizados
                    if self.carrying.color == 'yellow' and not self.rewards_history['got_yellow']:
                        reward += 10.0; self.rewards_history['got_yellow'] = True
                    elif self.carrying.color == 'red' and not self.rewards_history['got_red']:
                        reward += 10.0; self.rewards_history['got_red'] = True
                    elif self.carrying.color == 'blue' and not self.rewards_history['got_blue']:
                        reward += 10.0; self.rewards_history['got_blue'] = True

        # 4. ACCIÓN ABRIR (TOGGLE) - CONSUME LLAVE
        elif action == 5:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell and fwd_cell.type == 'door' and fwd_cell.is_locked:
                if self.carrying and self.carrying.color == fwd_cell.color:
                    fwd_cell.is_locked = False
                    fwd_cell.is_open = True
                    self.carrying = None # <--- ADIÓS LLAVE
                    
                    # Tus rewards personalizados
                    if fwd_cell.color == 'yellow' and not self.rewards_history['opened_yellow']:
                        reward += 10.0; self.rewards_history['opened_yellow'] = True
                    elif fwd_cell.color == 'red' and not self.rewards_history['opened_red']:
                        reward += 10.0; self.rewards_history['opened_red'] = True
                    elif fwd_cell.color == 'blue' and not self.rewards_history['opened_blue']:
                        reward += 10.0; self.rewards_history['opened_blue'] = True

        # 5. REWARD SHAPING (TU LÓGICA)
        new_target_pos = self._get_target_pos()
        
        # Si el objetivo cambió (ej: cogiste llave, ahora ve a puerta), reseteamos la distancia
        if not np.array_equal(np.array(old_target_pos), np.array(new_target_pos)):
            self.prev_dist = np.abs(np.array(self.agent_pos) - np.array(new_target_pos)).sum()
        
        curr_dist = np.abs(np.array(self.agent_pos) - np.array(new_target_pos)).sum()
        reward += (self.prev_dist - curr_dist) * 0.1
        self.prev_dist = curr_dist

        # 6. VICTORIA
        if self.grid.get(*self.agent_pos) is not None and self.grid.get(*self.agent_pos).type == 'goal':
            terminated = True
            reward += 50.0

        if self.step_count >= self.max_steps:
            truncated = True

        obs = self.gen_obs()
        return obs, reward, terminated, truncated, {}

# REGISTRO
gym.register(
    id='MiniGrid-Curriculum-Arcade-v0',
    entry_point='custom_env:CurriculumTempleEnv'
)