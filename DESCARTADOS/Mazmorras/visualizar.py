import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from sb3_contrib import RecurrentPPO
import time
from minigrid.core.world_object import Door, Goal, Key, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv
from gymnasium.envs.registration import register
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace

# 1. Registrar el entorno (si lo haces en un script aparte)
# Asegúrate de que la clase CurriculumTempleEnv esté definida o importada

class CurriculumTempleEnv(MiniGridEnv):
    def __init__(self, render_mode=None):
        self.grid_w = 19
        self.grid_h = 19
        
        # --- NIVEL DE CURRICULUM ---
        # 1: Final Room (Solo Meta)
        # 2: Blue Key Room (Llave Azul -> Puerta Azul -> Meta)
        # 3: Red Key Room (Llave Roja -> P. Roja -> Llave Azul -> P. Azul -> Meta)
        # 4: Full Run (Amarilla -> Roja -> Azul -> Meta)
        self.current_level = 1 
        
        mission_space = MissionSpace(mission_func=lambda: "traverse the temple")
        super().__init__(
            mission_space=mission_space,
            width=self.grid_w,
            height=self.grid_h,
            max_steps=2000, 
            render_mode=render_mode
        )

    def set_level(self, level):
        """Método para actualizar el nivel desde el Callback"""
        # Solo imprimimos si cambia el nivel para no saturar la consola
        if self.current_level != level:
            print(f"--- Environment switching to Level {level} ---")
            self.current_level = level

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # --- PAREDES ESTRUCTURALES ---
        self.grid.vert_wall(6, 0); self.grid.vert_wall(12, 0)
        self.grid.horz_wall(0, 6); self.grid.horz_wall(0, 12)

        # --- HUECOS ---
        self.grid.set(6, 3, None); self.grid.set(12, 3, None) 
        self.grid.set(12, 9, None); self.grid.set(6, 9, None) 
        self.grid.set(6, 15, None) 

        # --- OBJETOS ---
        # Puertas
        self.door_yellow = Door('yellow', is_locked=True)
        self.grid.set(15, 6, self.door_yellow) 
        self.door_red = Door('red', is_locked=True)
        self.grid.set(3, 12, self.door_red)
        self.door_blue = Door('blue', is_locked=True)
        self.grid.set(12, 15, self.door_blue)

        # Llaves
        self.key_yellow = Key('yellow'); self.place_obj(self.key_yellow, top=(7, 1), size=(5, 5))
        self.key_red = Key('red'); self.place_obj(self.key_red, top=(7, 7), size=(5, 5))
        self.key_blue = Key('blue'); self.place_obj(self.key_blue, top=(7, 13), size=(5, 5))

        # Meta
        # Colocamos la meta. place_obj busca un hueco libre en la zona definida.
        self.place_obj(Goal(), top=(13, 13), size=(5, 5))
        # Nota: La posición exacta puede variar dentro de la sala 13,13, pero la lógica de distancia usa el centro aprox.
        self.goal_pos = np.array([15, 15]) 

        # --- LAVA (Peligros) - LÓGICA CORREGIDA ---
        # Solo en el 3x3 central de cada habitación para dejar pasillos seguros pegados a las paredes.
        lava_density = 0.3 # Aumentamos un poco la densidad ya que el área es menor
        
        # Iteramos sobre las 9 habitaciones (3 columnas x 3 filas)
        for col_room in range(3):
            for row_room in range(3):
                # Coordenada base (esquina superior izquierda de las paredes de la habitación)
                base_x = col_room * 6
                base_y = row_room * 6
                
                # El interior de la habitación va de base+1 a base+5.
                # El CENTRO 3x3 va de base+2 a base+4 (indices 2, 3, 4 relativos a la pared).
                for x in range(base_x + 2, base_x + 5):
                    for y in range(base_y + 2, base_y + 5):

                        if col_room == 2 and row_room == 2:
                            continue # Saltamos esta habitación, no ponemos lava aquí
                        
                        # Verificamos que no haya nada (ni puertas, ni llaves, ni meta ya colocada)
                        if self.grid.get(x, y) is None:
                            if self.np_random.uniform(0, 1) < lava_density:
                                self.grid.set(x, y, Lava())

        # --- SPAWN LOGIC SEGÚN CURRICULUM ---
        if self.current_level == 1:
            # FASE 1: Sala de Victoria (Abajo-Derecha)
            self.place_agent(top=(13, 13), size=(5, 5))
            
        elif self.current_level == 2:
            # FASE 2: Sala Llave Azul (Abajo-Izquierda)
            self.place_agent(top=(1, 13), size=(5, 5))
            
        elif self.current_level == 3:
            # FASE 3: Sala Llave Roja (Medio-Derecha)
            self.place_agent(top=(13, 7), size=(5, 5))
            
        else: # Level 4
            # FASE 4: Full Hardcore (Arriba-Izquierda)
            self.place_agent(top=(1, 1), size=(5, 5))

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.rewards_history = {
            'got_yellow': False, 'opened_yellow': False,
            'got_red': False, 'opened_red': False,
            'got_blue': False, 'opened_blue': False
        }
        self.prev_dist = np.abs(self.agent_pos - self.goal_pos).sum()
        return obs, info

    def step(self, action):
        pre_carrying = self.carrying
        pre_yellow_open = self.door_yellow.is_open
        pre_red_open = self.door_red.is_open
        pre_blue_open = self.door_blue.is_open

        obs, reward, terminated, truncated, info = super().step(action)

        # --- RECOMPENSAS SOLICITADAS ---
        
        # 1. SHAPING POR DISTANCIA (+- 0.1)
        curr_dist = np.abs(self.agent_pos - self.goal_pos).sum()
        dist_diff = self.prev_dist - curr_dist 
        
        if dist_diff > 0:
            reward += 0.1
        elif dist_diff < 0:
            reward -= 0.1
        
        self.prev_dist = curr_dist

        # 2. VICTORIA (+8)
        if terminated and self.grid.get(*self.agent_pos) is not None and self.grid.get(*self.agent_pos).type == 'goal':
            reward += 8.0 

        # 3. LLAVES (+3)
        if pre_carrying != self.key_yellow and self.carrying == self.key_yellow:
            if not self.rewards_history['got_yellow']: 
                reward += 3.0; self.rewards_history['got_yellow'] = True

        if pre_carrying != self.key_red and self.carrying == self.key_red:
            if not self.rewards_history['got_red']: 
                reward += 3.0; self.rewards_history['got_red'] = True

        if pre_carrying != self.key_blue and self.carrying == self.key_blue:
            if not self.rewards_history['got_blue']: 
                reward += 3.0; self.rewards_history['got_blue'] = True

        # 4. PUERTAS (+3)
        if not pre_yellow_open and self.door_yellow.is_open:
            if not self.rewards_history['opened_yellow']: 
                reward += 3.0; self.rewards_history['opened_yellow'] = True

        if not pre_red_open and self.door_red.is_open:
            if not self.rewards_history['opened_red']: 
                reward += 3.0; self.rewards_history['opened_red'] = True

        if not pre_blue_open and self.door_blue.is_open:
            if not self.rewards_history['opened_blue']: 
                reward += 3.0; self.rewards_history['opened_blue'] = True

        return obs, reward, terminated, truncated, info

register(id='MiniGrid-Curriculum-v0', entry_point='__main__:CurriculumTempleEnv')

def visualize_level_2(model_path):
    # Crear el entorno en modo 'human' para ver la ventana
    env = gym.make('MiniGrid-Curriculum-v0', render_mode="human")
    env = ImgObsWrapper(env)
    
    # Forzar el nivel 2
    env.unwrapped.set_level(2)
    
    # Cargar el modelo
    model = RecurrentPPO.load(model_path)
    
    obs, _ = env.reset()
    
    # RecurrentPPO necesita mantener el estado de la memoria (LSTM)
    lstm_states = None
    episode_starts = [True]

    try:
        while True:
            # El modelo predice la acción basada en la observación y el estado de la memoria
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts, 
                deterministic=True # Determinista para ver su mejor comportamiento
            )
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_starts = [terminated or truncated]
            
            env.render()
            time.sleep(0.1) # Pausa para que el ojo humano pueda seguirlo

            if terminated or truncated:
                print(f"Episodio terminado. Recompensa: {reward}")
                obs, _ = env.reset()
                lstm_states = None # Resetear memoria de la LSTM
                episode_starts = [True]
                
    except KeyboardInterrupt:
        print("Cerrando visualización...")
        env.close()

if __name__ == "__main__":
    # Cambia esto por la ruta a tu modelo guardado
    PATH = "./Curriculum/Models/Curriculum_Interrupted.zip" 
    visualize_level_2(PATH)