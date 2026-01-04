import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
import gymnasium as gym
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Key, Door, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace

class InfiniteRoomsEnv(MiniGridEnv):
    def __init__(self, num_rooms=100, max_steps=1000, **kwargs):
        self.num_rooms = num_rooms
        # Calculamos el tamaño total: cada habitación es 6x6, pero comparten paredes
        # Un pasillo de habitaciones hacia la derecha
        grid_size = 6 * num_rooms 
        
        mission_space = MissionSpace(mission_func=lambda: "Avanza lo mas lejos posible")
        
        super().__init__(
            mission_space=mission_space,
            grid_size=grid_size,
            max_steps=max_steps,
            see_through_walls=False,
            **kwargs
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, 6) # Techo y suelo

        for i in range(self.num_rooms):
            x_offset = i * 5 # Las habitaciones comparten la pared lateral
            
            # Crear pared derecha de la habitación actual (excepto la última)
            if i < self.num_rooms - 1:
                for y in range(6):
                    self.grid.set(x_offset + 5, y, Wall())
                
                # Añadir puerta o obstáculo
                self._add_challenge(i, x_offset)

        # Agente al inicio
        self.agent_pos = (1, 3)
        self.agent_dir = 0

    def _add_challenge(self, room_idx, x_offset):
        y_door = 3
        # Definir los límites internos de la habitación actual (6x6 incluyendo paredes)
        # El interior es de x=(x_offset + 1) a (x_offset + 4) y y=1 a 4
        
        if room_idx % 2 == 0:  # Habitaciones pares: Lava
            # Colocamos lava en lugares fijos o podrías aleatorizarlos también
            self.grid.set(x_offset + 3, 2, Lava())
            self.grid.set(x_offset + 3, 4, Lava())
            self.grid.set(x_offset + 5, y_door, None)  # Hueco libre
        else:  # Habitaciones impares: Puerta y Llave Aleatoria
            color = self._rand_elem(['red', 'green', 'blue'])
            self.grid.set(x_offset + 5, y_door, Door(color, is_locked=True))
            
            # --- Lógica de la Llave Aleatoria ---
            # Intentamos encontrar una posición libre (None) en la habitación
            key_placed = False
            while not key_placed:
                # Generar coordenadas dentro del área 4x4 interna de la habitación
                kx = self.np_random.integers(x_offset + 1, x_offset + 5)
                ky = self.np_random.integers(1, 5)
                
                # Solo poner la llave si la celda está vacía
                if self.grid.get(kx, ky) is None:
                    self.grid.set(kx, ky, Key(color))
                    key_placed = True

    def step(self, action):
        # 1. Estado previo para comparar
        old_x = self.agent_pos[0]
        was_carrying = self.carrying is not None
        
        # 2. Ejecutar la acción
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 3. Inicializar recompensa personalizada
        # (Sobreescribimos la de MiniGrid que suele ser 0 si no hay Meta)
        custom_reward = 0 

        # --- Recompensa por movimiento (X) ---
        new_x = self.agent_pos[0]
        if new_x > old_x:
            custom_reward += 0.1  # Avanzar
        elif new_x < old_x:
            custom_reward -= 0.1  # Retroceder

        # --- Recompensa por habitación nueva ---
        # Cada habitación mide 5 de ancho efectivo (6 menos la pared compartida)
        if new_x // 5 > old_x // 5:
            custom_reward += 0.5
            print(f"¡Habitación {new_x // 5} alcanzada!")

        # --- Recompensa por Objetos ---
        # Coger llave
        if self.carrying is not None and not was_carrying:
            custom_reward += 1.0
            
        # Abrir puerta (detectamos si la acción fue 'abrir' y si hubo éxito)
        # En MiniGrid, la acción 4 es 'Toggle' (abrir/cerrar)
        if action == self.actions.toggle:
            # Si el agente ya no tiene la llave y estaba frente a la puerta, 
            # es que la ha usado para abrirla.
            if was_carrying and self.carrying is None:
                custom_reward += 2.0

        # --- Penalización por Muerte ---
        if terminated and reward <= 0: # Si termina sin llegar a una meta final
            custom_reward -= 10.0

        return obs, custom_reward, terminated, truncated, info
def record_agent():
    # 1. Crear entorno (usa render_mode="rgb_array" para grabar)
    env = InfiniteRoomsEnv(num_rooms=100, max_steps=5000, render_mode="rgb_array")
    
    # 2. Grabador de video (guardará en la carpeta 'videos')
    env = RecordVideo(env, video_folder="./videos", name_prefix="agente_experto")
    
    # 3. Wrapper de observaciones
    env = FlatObsWrapper(env)

    # 4. Cargar modelo
    model = PPO.load("ppo_infinite_navigator.zip")

    obs, _ = env.reset()
    for _ in range(5000): # Grabamos 5000 pasos de pura habilidad
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
            
    env.close()
    print("¡Video grabado en la carpeta /videos!")

if __name__ == "__main__":
    record_agent()