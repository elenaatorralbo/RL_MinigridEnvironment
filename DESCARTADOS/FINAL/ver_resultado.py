import gymnasium as gym
import torch
import numpy as np
import time
from minigrid.wrappers import ImgObsWrapper
from sb3_contrib import RecurrentPPO
from environment import SurvivalCorridorEnv

def main():
    print("Cargando entorno y agente...")
    
    # 1. Configuración idéntica al entrenamiento
    env = SurvivalCorridorEnv(render_mode="human", num_rooms=25, agent_view_size=9)
    env = ImgObsWrapper(env) # Wrapper visual imprescindible

    # 2. Cargar modelo
    try:
        model = RecurrentPPO.load("checkpoints_final/rl_model_final_2700896_steps.zip", env=env)
        print("¡Modelo cargado con éxito!")
    except:
        print("Error: No se encuentra el archivo 'agente_pacman_survival_lstm_final.zip'")
        return

    # 3. Bucle de visualización
    for episodio in range(1, 6):
        obs, _ = env.reset()
        
        # Inicializar memoria LSTM
        lstm_states = None 
        episode_starts = np.ones((1,), dtype=bool)
        
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        
        print(f"\n==================== INICIO EPISODIO {episodio} ====================")
        
        while not terminated and not truncated:
            # Predicción con memoria
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts,
                deterministic=False # False para ver algo de variedad, True para ver la mejor ruta estricta
            )
            
            episode_starts[0] = False
            
            # --- CORRECCIÓN CRÍTICA AQUÍ ---
            # Convertimos el array de numpy a un entero simple de Python
            if isinstance(action, np.ndarray):
                action = int(action.item())
            # -------------------------------
            
            # Ejecutar paso
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Renderizado
            env.render() 
            
            # Info por consola
            nombres_acciones = ["DERECHA", "ABAJO", "IZQUIERDA", "ARRIBA"]
            # Aseguramos que el índice sea válido por si acaso
            nombre_accion = nombres_acciones[action] if action < 4 else f"Acción {action}"
            
            print(f"[Paso {steps:03d}] Acción: {nombre_accion} | Reward: {reward:5.2f} | Total: {total_reward:6.2f}")
            
            time.sleep(0.005) # Velocidad de visualización

        print(f"==================== FIN EPISODIO {episodio} (Reward Total: {total_reward:.2f}) ====================")
        time.sleep(1)

    env.close()

if __name__ == "__main__":
    main()