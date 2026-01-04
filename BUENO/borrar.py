import gymnasium as gym
import minigrid
import time

def visualizar_doorkey():
    env_id = "MiniGrid-DoorKey-8x8-v0"
    
    print(f"\n--- 🔑 VISUALIZANDO: {env_id} ---")
    
    try:
        # render_mode="human" es lo que hace que se abra la ventana emergente
        env = gym.make(env_id, render_mode="human")
        
        # Reseteamos para generar el nivel
        obs, _ = env.reset()
        
        print("📸 Se ha abierto la ventana.")
        print("⏳ Tienes 10 segundos para hacer la captura de pantalla...")
        
        # Mantenemos la ventana abierta durante 10 segundos (100 frames * 0.1s)
        for i in range(100):
            env.render()
            # Si quieres que el agente se mueva aleatoriamente para ver animaciones,
            # descomenta la siguiente línea:
            # env.step(env.action_space.sample()) 
            time.sleep(0.1)
            
        env.close()
        print("✅ Visualización finalizada.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Asegúrate de tener instalado 'gymnasium' y 'minigrid'.")

if __name__ == "__main__":
    visualizar_doorkey()