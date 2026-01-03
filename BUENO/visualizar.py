import gymnasium as gym
import minigrid
from stable_baselines3 import PPO
from gymnasium import spaces, ObservationWrapper
import os

# =============================================================================
# 1. WRAPPER (EL MISMO DE SIEMPRE)
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
# 2. VISUALIZADOR MODO MARATÓN
# =============================================================================
def ver_maraton():
    
    # --- RUTA DEL MODELO ---
    # Opción A: El modelo final (si acabó el entrenamiento)
    model_name = "checkpoints/Nivel_2_5_Intermedio/Nivel_2_5_Intermedio_500000_steps.zip"
    
    # Opción B: Un checkpoint específico (si lo paraste a medias)
    # model_name = "checkpoints/MultiRoom_Expert_N20/MultiRoom_Expert_N20_1000000_steps.zip"

    if not os.path.exists(model_name):
        print(f"❌ ERROR: No encuentro {model_name}")
        print("Asegúrate de que el entrenamiento ha terminado o ajusta la ruta.")
        return

    # --- CONFIGURACIÓN DEL ENTORNO GIGANTE ---
    env_id = "MiniGrid-MultiRoom-N6-v0"
    
    # ¡ESTO ES LO IMPORTANTE! 
    # Le decimos a Minigrid que ignore el "N6" y cree 15-20 habitaciones.
    config_maraton = {
        "minNumRooms": 15,
        "maxNumRooms": 16,
        "maxRoomSize": 8
    }

    print(f"--- CARGANDO MODO MARATÓN ---")
    print(f"Configuración: {config_maraton}")
    
    # Creamos el entorno con render_mode="human" para verlo
    # El ancho de la ventana se ajustará automáticamente, pero el mapa será ENORME.
    env = gym.make(env_id, render_mode="human", **config_maraton)
    env = ImgObsWrapper(env)

    print(f"🧠 Cargando modelo: {model_name}...")
    try:
        model = PPO.load(model_name)
    except Exception as e:
        print(f"Error al cargar: {e}")
        return

    print("\n✅ ¡LISTO! Prepárate para ver un laberinto gigante.")
    print("   El agente puede tardar un poco en cruzarlo (son muchas habitaciones).")

    obs, _ = env.reset()
    
    try:
        while True:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            env.render()
            
            if terminated or truncated:
                if reward > 0:
                    print("🏆 ¡MARATÓN COMPLETADO! El agente llegó a la meta.")
                else:
                    print("💀 Se acabó el tiempo (es normal en mapas tan grandes al principio).")
                
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("\nCerrando...")
        env.close()

if __name__ == "__main__":
    ver_maraton()