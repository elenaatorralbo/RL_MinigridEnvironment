import pickle
import time
import gymnasium as gym
import numpy as np
from env import SequentialRoomsEnv
from train import SimpleRLAgent

# Diccionario para traducir acciones de MiniGrid a texto
ACTION_NAMES = {
    0: "⬅️ Girar Izquierda",
    1: "➡️ Girar Derecha",
    2: "⬆️ Avanzar",
    3: "✋ Coger Objeto (Pickup)",
    4: "⬇️ Soltar Objeto (Drop)",
    5: "🔑 Interactuar/Abrir (Toggle)",
    6: "✅ Hecho (Done)"
}

def visualize(model_file, start_room, epsilon=0.1):
    env = SequentialRoomsEnv(render_mode="human", start_room_index=start_room)
    agent = SimpleRLAgent(env.action_space.n)
    
    print(f"\n--- CARGANDO MODELO: {model_file} ---")
    try:
        with open(model_file, "rb") as f:
            agent.q_table = pickle.load(f)
        print("✅ Modelo cargado correctamente.")
    except FileNotFoundError:
        print(f"❌ ERROR: No se encuentra el archivo '{model_file}'.")
        return

    # Configuramos el nivel de exploración (ruido)
    agent.epsilon = epsilon 

    obs, _ = env.reset()
    env.render()
    
    terminated = False
    truncated = False
    total_reward = 0
    steps = 0
    
    print(f"\n--- INICIANDO VISUALIZACIÓN (Room {start_room}) ---")
    print("Pulsa Ctrl+C para detener.\n")
    time.sleep(1)
    
    try:
        while not (terminated or truncated):
            # 1. El agente decide acción
            action = agent.get_action(obs)
            
            # 2. Imprimir la acción por consola
            action_name = ACTION_NAMES.get(action, f"Desconocida ({action})")
            print(f"Paso {steps:03d} | Acción: {action_name}", end="")
            
            # 3. Ejecutar paso
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            # Imprimir recompensa si ocurre algo interesante
            if reward > 0:
                print(f" | 🎉 Reward: +{reward:.2f}", end="")
            elif reward < -0.05: # Si es un castigo grande (lava)
                print(f" | 💀 Reward: {reward:.2f}", end="")
            
            print("") # Salto de línea
            
            # 4. Renderizar
            env.render()
            
            # Ajusta la velocidad aquí
            time.sleep(0.1) 

        print(f"\n--- FIN DEL EPISODIO ---")
        print(f"Pasos totales: {steps}")
        print(f"Recompensa Final: {total_reward:.2f}")
        
        if total_reward > 0:
            print("Resultado: ¡VICTORIA! 🏆")
        else:
            print("Resultado: Derrota o Muerte 💀")
            
    except KeyboardInterrupt:
        print("\nVisualización detenida por el usuario.")
    finally:
        time.sleep(2)
        env.close()

if __name__ == "__main__":
    # --- CONFIGURACIÓN ---
    ARCHIVO_MODELO = "q_table_level_19.pkl"  # O el archivo que quieras probar
    HABITACION_INICIO = 19
    RUIDO = 0.1  # 10% de aleatoriedad para evitar bucles
    
    visualize(ARCHIVO_MODELO, HABITACION_INICIO, RUIDO)