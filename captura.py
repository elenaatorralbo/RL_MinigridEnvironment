import gymnasium as gym
from gymnasium.envs.registration import register
from minigrid.envs.multiroom import MultiRoomEnv
from minigrid.core.world_object import Door, Key
import random
import os
import matplotlib.pyplot as plt  # Necesario para guardar las imágenes


# =============================================================================
# 1. DEFINICIÓN DEL ENTORNO (Copiado de tu script original)
# =============================================================================
class CorredorMulticolor(MultiRoomEnv):
    def __init__(self, n_rooms=4, key_prob=0.2, **kwargs):
        super().__init__(
            minNumRooms=n_rooms,
            maxNumRooms=n_rooms,
            maxRoomSize=10,
            **kwargs
        )
        self.key_prob = key_prob

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        valid_colors = ['red', 'blue', 'purple', 'yellow', 'grey']

        for i, room in enumerate(self.rooms):
            if i == len(self.rooms) - 1:
                break

            if random.random() < self.key_prob:
                door_pos = room.exitDoorPos
                color = random.choice(valid_colors)

                self.grid.set(door_pos[0], door_pos[1], Door(color, is_locked=True))

                self.place_obj(
                    Key(color),
                    top=room.top,
                    size=room.size,
                    max_tries=100
                )


# Registramos el entorno si no existe
if "MiniGrid-CorredorMulticolor-v0" in gym.envs.registry:
    del gym.envs.registry["MiniGrid-CorredorMulticolor-v0"]

register(
    id="MiniGrid-CorredorMulticolor-v0",
    entry_point=__name__ + ":CorredorMulticolor",
)


# =============================================================================
# 2. FUNCIÓN PARA GENERAR CAPTURAS
# =============================================================================
def capturar_entornos():
    # Las fases que definiste en tu curriculum
    stages = [3, 6, 9, 12]

    # Carpeta donde se guardarán las fotos
    output_dir = "capturas_entornos"
    os.makedirs(output_dir, exist_ok=True)

    print(f"📸 Iniciando proceso de captura de entornos...")
    print(f"📂 Las imágenes se guardarán en la carpeta: ./{output_dir}/")

    for n_rooms in stages:
        # Creamos el entorno con render_mode='rgb_array' para obtener la imagen
        env = gym.make("MiniGrid-CorredorMulticolor-v0", render_mode="rgb_array", n_rooms=n_rooms)

        # Reseteamos para generar el mapa aleatorio
        env.reset()

        # Obtenemos la imagen (array de píxeles)
        img = env.render()

        # Nombre del archivo
        filename = f"{output_dir}/entorno_{n_rooms}_habitaciones.png"

        # Guardamos la imagen usando Matplotlib
        # Usamos origin='upper' porque las coordenadas de imagen suelen empezar arriba
        plt.imsave(filename, img)

        print(f"   ✅ Captura guardada: {filename}")

        env.close()

    print("\n✨ ¡Listo! Revisa la carpeta para ver tus mapas.")


if __name__ == "__main__":
    capturar_entornos()