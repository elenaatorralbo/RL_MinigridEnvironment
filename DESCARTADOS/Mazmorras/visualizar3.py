import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Key, Door, Wall, Lava
from minigrid.minigrid_env import MiniGridEnv

# ==============================================================================
# TU ENTORNO (Copiado exactamente de la última versión acordada)
# ==============================================================================
class CurriculumTempleEnv(MiniGridEnv):
    def __init__(self, render_mode=None, max_steps=1000):
        self.grid_w = 19
        self.grid_h = 19
        self.current_level = 1 
        
        self.rewards_history = {
            'got_yellow': False, 'opened_yellow': False, 
            'got_red': False, 'opened_red': False, 
            'got_blue': False, 'opened_blue': False
        }
        
        self.visited_rooms = set()
        
        mission_space = MissionSpace(mission_func=lambda: "traverse the temple")
        
        # NOTA: Para visualización, usaremos 'rgb_array' y matplotlib
        super().__init__(
            mission_space=mission_space,
            width=self.grid_w,
            height=self.grid_h,
            max_steps=max_steps, 
            render_mode=render_mode
        )
        
        self.action_space = spaces.Discrete(6)

    def set_level(self, level):
        # Simplificado para visualización, solo cambia la variable interna
        self.current_level = level

    def reset(self, *, seed=None, options=None):
        self.rewards_history = {k: False for k in self.rewards_history}
        obs, info = super().reset(seed=seed, options=options)
        
        self.visited_rooms = set()
        start_room = (self.agent_pos[0] // 6, self.agent_pos[1] // 6)
        self.visited_rooms.add(start_room)
        
        return obs, info

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.grid.vert_wall(6, 0); self.grid.vert_wall(12, 0)
        self.grid.horz_wall(0, 6); self.grid.horz_wall(0, 12)

        # FASE 1: Zig-Zag (Spawn Fijo Lejano)
        if self.current_level == 1:
            self.grid.set(6, 3, None); self.grid.set(12, 3, None) 
            self.grid.set(12, 9, None); self.grid.set(6, 9, None) 
            self.grid.set(6, 15, None) 
            
            # SPAWN FIJO ARRIBA A LA IZQUIERDA
            self.place_agent(top=(1, 1), size=(5, 5)) 

        # FASE 2: PUZZLE
        if self.current_level >= 2:
            self.door_yellow = Door('yellow', is_locked=True); self.grid.set(15, 6, self.door_yellow) 
            self.door_red = Door('red', is_locked=True); self.grid.set(3, 12, self.door_red)
            self.door_blue = Door('blue', is_locked=True); self.grid.set(12, 15, self.door_blue)

            self.key_yellow = Key('yellow'); self.place_obj(self.key_yellow, top=(7, 1), size=(5, 5))
            self.key_red = Key('red'); self.place_obj(self.key_red, top=(7, 7), size=(5, 5))
            self.key_blue = Key('blue'); self.place_obj(self.key_blue, top=(7, 13), size=(5, 5))
            
            self.grid.set(6, 3, None); self.grid.set(12, 3, None) 
            # También forzamos spawn difícil aquí
            self.place_agent(top=(1, 1), size=(5, 5))

        # FASE 3: LAVA
        if self.current_level >= 3:
            lava_density = 0.25
            for col_room in range(3):
                for row_room in range(3):
                    if col_room == 2 and row_room == 2: continue 
                    base_x = col_room * 6; base_y = row_room * 6
                    for x in range(base_x + 2, base_x + 5):
                        for y in range(base_y + 2, base_y + 5):
                            if self.grid.get(x, y) is None:
                                if self.np_random.uniform(0, 1) < lava_density:
                                    self.grid.set(x, y, Lava())

        self.place_obj(Goal(), top=(13, 13), size=(5, 5))
        self.mission = "traverse the temple"

    # (Los métodos step y _get_target_pos no hacen falta para solo visualizar el mapa inicial)

# ==============================================================================
# LÓGICA DE VISUALIZACIÓN
# ==============================================================================
if __name__ == "__main__":
    # Creamos el entorno en modo 'rgb_array' para que matplotlib pueda dibujarlo
    env = CurriculumTempleEnv(render_mode='rgb_array')

    # Preparamos la ventana gráfica
    plt.ion() # Modo interactivo
    fig, ax = plt.subplots(figsize=(8, 8))
    
    print("\n--- INICIANDO VISUALIZACIÓN DE FASES ---")
    print("Mira la ventana gráfica que se ha abierto.")

    fases_a_revisar = [1, 2, 3]

    for fase in fases_a_revisar:
        print(f"\n[GENERANDO FASE {fase}]...")
        
        # 1. Establecemos el nivel
        env.set_level(fase)
        
        # 2. Reseteamos para que se genere el mapa con las reglas de esa fase
        env.reset()
        
        # 3. Obtenemos la imagen del mapa
        img = env.render()
        
        # 4. La mostramos en la ventana
        ax.clear()
        ax.imshow(img)
        ax.set_title(f"VISUALIZACIÓN: FASE {fase}", fontsize=16)
        ax.axis('off') # Quitar ejes para que se vea más limpio
        plt.draw()
        plt.pause(0.1) # Pequeña pausa para asegurar que se dibuja
        
        # 5. Descripción de lo que deberías ver
        if fase == 1:
            print(">> DEBERÍAS VER: Agente arriba-izquierda (triángulo rojo). Meta abajo-derecha (cuadrado verde). HUECOS abiertos en las paredes para hacer zig-zag.")
        elif fase == 2:
            print(">> DEBERÍAS VER: Igual que Fase 1, pero ahora hay PUERTAS de colores cerradas y LLAVES de colores repartidas por el mapa.")
        elif fase == 3:
            print(">> DEBERÍAS VER: Igual que Fase 2, pero ahora el suelo tiene manchas naranjas de LAVA mortal.")

        # 6. Esperar al usuario
        input(f"\n[ Pulsa ENTER en esta terminal para ver la siguiente fase... ]")

    print("\n--- Visualización completada. Cerrando. ---")
    plt.close()
    env.close()