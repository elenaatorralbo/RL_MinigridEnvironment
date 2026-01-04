import gym
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# IMPORTANTE: Importa aquí tu entorno si es una clase personalizada
# from mazmorraCOMPLETA import TuEntornoPersonalizado 
# O si usas gym.make con un ID registrado, asegúrate de usar el mismo ID.

def probar_modelo(model_path, env_id, num_episodes=5, render=True):
    """
    Carga un modelo y lo ejecuta en el entorno.
    """
    print(f"--- Cargando modelo desde: {model_path} ---")
    
    # 1. Cargar el modelo
    # Asegúrate de usar el device correcto (cpu suele ser mejor para inferencia visual simple)
    model = PPO.load(model_path, device="cpu")

    # 2. Crear el entorno
    # Si tu entorno requiere configuración especial (wrappers), añádelos aquí
    # Usa render_mode='human' para ver la ventana del juego
    try:
        env = gym.make(env_id, render_mode='human' if render else None)
    except:
        # Fallback para versiones antiguas de gym
        env = gym.make(env_id)

    print("--- Iniciando pruebas visuales ---")
    
    for episode in range(1, num_episodes + 1):
        obs = env.reset()
        done = False
        score = 0
        steps = 0
        
        while not done:
            # El modelo predice la acción. deterministic=True es MEJOR para testear
            action, _state = model.predict(obs, deterministic=True)
            
            # Ejecutar paso
            # Nota: Gym nuevo devuelve 5 valores, el viejo 4. Esto lo maneja:
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            
            score += reward
            steps += 1
            
            if render:
                env.render()
                time.sleep(0.05) # Pequeña pausa para que puedas verlo bien

        print(f"Episodio {episode}: Pasos = {steps}, Recompensa Total = {score:.2f}")
        
        # Análisis rápido de Reward Hacking
        if steps >= 200 and score > 0:
            print("  ⚠️ ALERTA: El agente agotó el tiempo pero ganó recompensa.")
        elif steps < 200 and score > 0:
            print("  ✅ ÉXITO: El agente resolvió el nivel.")
            
    env.close()

def evaluar_rendimiento_puro(model_path, env_id):
    """
    Evalúa matemáticamente sin renderizar (más rápido y preciso).
    """
    print("\n--- Iniciando evaluación estadística (10 episodios) ---")
    env = gym.make(env_id)
    env = Monitor(env) # Necesario para evaluar
    model = PPO.load(model_path, device="cpu")
    
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    
    print(f"Recompensa Media: {mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == "__main__":
    # --- CONFIGURACIÓN ---
    # 1. Pon aquí el nombre de tu entorno Minigrid (ej: 'MiniGrid-Empty-8x8-v0')
    # O el nombre con el que registraste tu mazmorra en mazmorraCOMPLETA.py
    ENV_NAME = "MiniGrid-Empty-8x8-v0" 
    
    # 2. Pon aquí la ruta a tu archivo .zip guardado
    MODEL_PATH = "modelos_guardados/modelo_FINAL_mazmorra.zip" # <--- CAMBIA ESTO
    
    # Ejecutar prueba visual
    probar_modelo(MODEL_PATH, ENV_NAME, num_episodes=3, render=True)
    
    # Ejecutar prueba estadística
    # evaluar_rendimiento_puro(MODEL_PATH, ENV_NAME)