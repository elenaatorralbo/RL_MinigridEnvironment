import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from minigrid.wrappers import FlatObsWrapper
# Importa aquí tu clase si la tienes en otro archivo, ej: from mi_entorno import CustomDoorKeyEnv
from env import CustomDoorKeyEnv

# --- 1. Preparación del Entorno ---
# Definimos una función que crea y "envuelve" el entorno.
# FlatObsWrapper es CRUCIAL: Convierte la visión compleja del robot (diccionario)
# en un simple array de números que la IA procesa mucho más rápido.
def make_env():
    env = CustomDoorKeyEnv(render_mode="rgb_array", size=10)
    env = FlatObsWrapper(env)
    return env

# Creamos el entorno vectorizado (opcionalmente podrías usar dummy wrappers)
env = make_env()

# --- 2. Crear el Modelo (El cerebro) ---
# Usamos PPO. 
# "MlpPolicy": Indica que usaremos una red neuronal estándar (no convolucional)
# porque FlatObsWrapper ya aplanó la imagen.
print("Iniciando entrenamiento... esto puede tardar unos segundos/minutos.")
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

# --- 3. Entrenar a la IA ---
# 100,000 pasos suele ser suficiente para este mapa simple.
# Verás el "mean_reward" subir en la consola.
model.learn(total_timesteps=100000)
print("¡Entrenamiento finalizado!")

# Guardamos el modelo entrenado
model.save("ppo_minigrid_doorkey")

# --- 4. Ver el resultado (Inferencia) ---
print("Mostrando al agente entrenado en acción...")

# Recreamos el entorno para visualización (render_mode="human")
eval_env = CustomDoorKeyEnv(render_mode="human", size=10)
eval_env = FlatObsWrapper(eval_env) # ¡Importante usar el mismo wrapper!

obs, _ = eval_env.reset()

for _ in range(100): # 100 pasos de demostración
    # El modelo predice la mejor acción
    action, _states = model.predict(obs, deterministic=True)
    
    # Ejecutamos la acción
    obs, reward, terminated, truncated, info = eval_env.step(action)
    eval_env.render()
    
    if terminated or truncated:
        print("Episodio terminado. Reiniciando...")
        obs, _ = eval_env.reset()

eval_env.close()