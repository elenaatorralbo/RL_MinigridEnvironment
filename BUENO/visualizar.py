import gymnasium as gym
from gymnasium.envs.registration import register
from minigrid.envs.multiroom import MultiRoomEnv
from minigrid.core.world_object import Door, Key, Lava, Goal, Wall
from minigrid.core.grid import Grid
import random
import time
from collections import deque

class CorredorLavaFinal(MultiRoomEnv):
    def __init__(self, n_rooms=4, key_prob=0.2, lava_prob=0.25, **kwargs):
        self.target_n_rooms = n_rooms
        self.key_prob = key_prob
        self.lava_prob = lava_prob
        
        super().__init__(
            minNumRooms=n_rooms, 
            maxNumRooms=n_rooms, 
            maxRoomSize=10, 
            **kwargs
        )

    def _gen_grid(self, width, height):
        max_retries = 500
        for _ in range(max_retries):
            
            # 1. Base
            self.grid = Grid(width, height)
            try:
                super()._gen_grid(width, height)
            except Exception:
                continue 
            
            # 2. Colocar Puertas y Llaves
            valid_colors = ['red', 'blue', 'purple', 'yellow', 'grey']
            for i, room in enumerate(self.rooms):
                if i == len(self.rooms) - 1: break 
                
                if random.random() < self.key_prob:
                    door_pos = room.exitDoorPos
                    color = random.choice(valid_colors)
                    self.grid.set(door_pos[0], door_pos[1], Door(color, is_locked=True))
                    # Colocamos la llave
                    self.place_obj(Key(color), top=room.top, size=room.size, max_tries=100)

            # 3. LAVA INTELIGENTE (Protege Puertas Y Llaves)
            self._add_lava_smart()

            # 4. Validación Final (BFS)
            if self._is_path_clear():
                return 

        print("⚠️ No se pudo generar nivel soluble. Reduciendo lava.")
        self.lava_prob = 0.05
        super()._gen_grid(width, height)

    def _add_lava_smart(self):
        """
        Escanea el mapa buscando Puertas y Llaves.
        Marca sus posiciones y adyacentes como 'SEGURAS'.
        Rellena el resto con lava al azar.
        """
        safe_cells = set()
        safe_cells.add(self.agent_pos)

        # A) Buscar Objetos Críticos en el mapa (Llaves y Puertas)
        # Recorremos todo el grid para encontrar dónde cayeron las llaves
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                obj = self.grid.get(x, y)
                
                # Si es Llave, Puerta o Meta -> Es sagrado
                if isinstance(obj, (Key, Door, Goal)):
                    safe_cells.add((x, y))
                    # Añadimos sus 4 vecinos a la lista segura (Breathing room)
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        safe_cells.add((x + dx, y + dy))

        # B) Añadir también las posiciones de puerta guardadas en los objetos Room
        # (A veces Minigrid borra el objeto puerta al fusionar, pero la pos es clave)
        for room in self.rooms:
            if room.entryDoorPos:
                safe_cells.add(room.entryDoorPos)
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    safe_cells.add((room.entryDoorPos[0]+dx, room.entryDoorPos[1]+dy))
            
            if room.exitDoorPos:
                safe_cells.add(room.exitDoorPos)
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    safe_cells.add((room.exitDoorPos[0]+dx, room.exitDoorPos[1]+dy))

        # C) Poner Lava donde NO sea seguro
        for room in self.rooms:
            for x in range(room.top[0], room.top[0] + room.size[0]):
                for y in range(room.top[1], room.top[1] + room.size[1]):
                    
                    # Verificar que la celda está vacía de objetos
                    cell = self.grid.get(x, y)
                    
                    # Si la celda está vacía Y NO está en la lista segura
                    if cell is None and (x, y) not in safe_cells:
                        if random.random() < self.lava_prob:
                            self.grid.set(x, y, Lava())

    def _is_path_clear(self):
        # BFS Básico: Solo comprueba conectividad física Start -> Goal
        # Gracias a _add_lava_smart, asumimos que si hay camino físico, 
        # las llaves son alcanzables porque están rodeadas de suelo seguro.
        start = self.agent_pos
        end = None
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                if isinstance(self.grid.get(i, j), Goal):
                    end = (i, j)
                    break
        if not end: return False 

        queue = deque([start])
        visited = set([start])

        while queue:
            x, y = queue.popleft()
            if (x, y) == end: return True

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                    if (nx, ny) not in visited:
                        cell = self.grid.get(nx, ny)
                        # Caminable: Vacío, Meta, Llave, Puerta (abierta o cerrada)
                        if not isinstance(cell, (Lava, Wall)):
                            visited.add((nx, ny))
                            queue.append((nx, ny))
        return False

# Registro
if "MiniGrid-VisualizerFinal-v0" in gym.envs.registry:
    del gym.envs.registry["MiniGrid-VisualizerFinal-v0"]

register(
    id="MiniGrid-VisualizerFinal-v0",
    entry_point=__name__ + ":CorredorLavaFinal",
)

def main():
    N_HABITACIONES = 12
    PROBABILIDAD_LAVA = 0.25 
    
    print(f"--- 🌋 VISUALIZANDO (LAVA FINAL: Protege Llaves y Puertas) 🌋 ---")
    
    env = gym.make(
        "MiniGrid-VisualizerFinal-v0", 
        render_mode="human", 
        n_rooms=N_HABITACIONES,
        lava_prob=PROBABILIDAD_LAVA
    )

    for i in range(3):
        print(f"Generando mapa {i+1}...")
        env.reset()
        for _ in range(100): 
            env.render()
            time.sleep(0.1)
    
    env.close()

if __name__ == "__main__":
    main()