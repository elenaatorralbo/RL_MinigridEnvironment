import gymnasium as gym
import time
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO

# Importamos las clases necesarias de tu archivo de entrenamiento
# Asegúrate de que el archivo se llame mazmorraINFINITA.py o cámbialo aquí
from mazmorraINFINITA import ContinuousDungeonEnv, DirectNavigationWrapper

def visualizar():
    # 1. Crear el entorno en modo humano
    # El render_mode "human" abrirá la ventana gráfica de MiniGrid
    env = ContinuousDungeonEnv(render_mode="human")
    env = DirectNavigationWrapper(env)
    env = FlatObsWrapper(env)

    # 2. Cargar el modelo guardado
    model_path = "modelo_infinito_final"  # Cambia esto por la ruta a tu modelo guardado
    
    try:
        # Cargamos el modelo y lo asociamos al entorno actual
        model = PPO.load(model_path, env=env)
        print(f"Modelo '{model_path}' cargado con éxito.")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    # 3. Bucle de ejecución
    obs, _ = env.reset()
    habitaciones_totales = 0
    
    print("\n--- Iniciando Simulación Continua ---")
    
    try:
        while True:
            # Predicción de la acción basada en la observación actual
            # deterministic=True asegura que el agente tome siempre la mejor opción aprendida
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Lógica para contar habitaciones en la visualización
            # Cada vez que el agente cruza una pared (x múltiplo de 6)
            if env.unwrapped.agent_pos[0] % 6 == 1 and action < 4:
                # Usamos un flag simple para no contar la misma habitación varias veces por paso
                habitaciones_totales = env.unwrapped.agent_pos[0] // 6
            
            # Renderizar el estado actual
            env.render()
            
            # Control de velocidad: 0.1 es un buen equilibrio para observar la lógica
            time.sleep(0.1)

            if terminated or truncated:
                print(f"💀 Fin del trayecto. Habitaciones recorridas: {habitaciones_totales}")
                obs, _ = env.reset()
                habitaciones_totales = 0
                print("Reiniciando mazmorra...")
                
    except KeyboardInterrupt:
        print("\nSimulación finalizada por el usuario.")
    finally:
        env.close()

if __name__ == "__main__":
    visualizar()