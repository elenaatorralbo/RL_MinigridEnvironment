import gymnasium as gym
import minigrid
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium import spaces, ObservationWrapper
import os

# =============================================================================
# 1. WRAPPER (EL MISMO)
# =============================================================================
class ImgObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        img_space = env.observation_space.spaces["image"]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=img_space.shape,
            dtype="uint8"
        )

    def observation(self, obs):
        return obs["image"]

# =============================================================================
# 2. ENTRENAMIENTO DE HABILIDAD "USAR LLAVES"
# =============================================================================
def train_doorkey():
    
    # Entorno oficial de Minigrid para aprender a usar llaves
    # Es simple: Coger llave -> Ir a puerta -> Abrir -> Salir
    env_id = "MiniGrid-DoorKey-8x8-v0"
    
    # Ruta de tu modelo experto en navegación (el del Nivel 2.5)
    prev_model_path = os.path.join("checkpoints", "Nivel_2_5_Intermedio", "Nivel_2_5_Intermedio_500000_steps.zip")
    new_model_name = "KeyDoor"

    print(f"--- 🔑 INICIANDO ENTRENAMIENTO DE LLAVES 🔑 ---")
    print(f"Objetivo: Aprender acciones 'Pickup' y 'Toggle'")

    # 1. Crear entorno DoorKey
    env = gym.make(env_id, render_mode=None)
    env = ImgObsWrapper(env)

    # 2. Cargar cerebro previo
    # NOTA: Esto es Transfer Learning puro. 
    # El agente sabe caminar, pero al principio intentará "chocar" contra la puerta
    # hasta que por suerte pulse "Espacio" (Toggle) o "Coger".
    if os.path.exists(prev_model_path):
        print(f"🧠 Cargando cerebro experto en navegación: {prev_model_path}")
        
        # Bajamos el Learning Rate para que no olvide cómo caminar mientras aprende a usar las manos
        custom_objects = {
            "learning_rate": 0.0001,
            "ent_coef": 0.02 # Subimos un poco la curiosidad para que pruebe botones nuevos
        }
        model = PPO.load(prev_model_path, env=env, custom_objects=custom_objects)
    else:
        print("⚠️ No encuentro el modelo anterior. Empezando de cero.")
        model = PPO("MlpPolicy", env, verbose=1)

    # 3. Guardado
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=f"./checkpoints/{new_model_name}/",
        name_prefix=new_model_name
    )

    # 4. Entrenar
    # 500k pasos deberían bastar para aprender la mecánica simple
    model.learn(
        total_timesteps=500_000, 
        callback=checkpoint_callback,
        reset_num_timesteps=True
    )
    
    model.save(new_model_name)
    print("--- 🔓 ENTRENAMIENTO DE LLAVES COMPLETADO ---")

if __name__ == "__main__":
    train_doorkey()