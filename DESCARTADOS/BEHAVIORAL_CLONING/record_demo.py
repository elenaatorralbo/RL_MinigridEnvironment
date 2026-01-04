import gymnasium as gym
import numpy as np
import pickle
import pygame
import time
from minigrid.wrappers import FullyObsWrapper, FlatObsWrapper

# --- IMPORTACIÓN Y REGISTRO MANUAL (PARA EVITAR ERRORES) ---
try:
    # Importamos la clase del entorno desde tu archivo custom_env.py
    from custom_env import CurriculumTempleEnv
    
    # Lo registramos AQUÍ MISMO para asegurar que Gym lo ve
    if 'MiniGrid-Curriculum-Arcade-v0' in gym.envs.registry:
        del gym.envs.registry['MiniGrid-Curriculum-Arcade-v0']
        
    gym.register(
        id='MiniGrid-Curriculum-Arcade-v0',
        entry_point='custom_env:CurriculumTempleEnv'
    )
    print("✅ Entorno 'MiniGrid-Curriculum-Arcade-v0' registrado correctamente.")
    
except ImportError:
    print("\n❌ ERROR CRÍTICO: No encuentro el archivo 'custom_env.py'.")
    print("Asegúrate de que 'custom_env.py' y 'record_demo.py' están en la misma carpeta.")
    exit()
# -----------------------------------------------------------

def record_demonstrations(num_episodes=20): 
    print(f"\n🎬 INICIANDO GRABACIÓN ARCADE - NIVEL 4 ({num_episodes} partidas)")
    print("---------------------------------------------------------------")
    
    # Creamos el entorno
    try:
        env = gym.make('MiniGrid-Curriculum-Arcade-v0', render_mode='human')
    except Exception as e:
        print(f"❌ Error al crear entorno: {e}")
        return

    # Forzamos Nivel 4 (Llaves + Lava)
    print("🔓 Configurando Nivel 4...")
    env.unwrapped.set_level(4)
    
    env = FullyObsWrapper(env)
    env = FlatObsWrapper(env)
    
    data = [] 
    
    print("\n🎮 CONTROLES NUEVOS (ARCADE):")
    print("  [⬅️ ⬆️ ⬇️ ➡️] : Movimiento Directo")
    print("  [ A ]      : Coger Llave (Pickup)")
    print("  [ESPACIO]  : Abrir Puerta (Toggle)")
    print("  [ESC]      : Salir y Guardar lo que lleves")

    successful_episodes = 0

    while successful_episodes < num_episodes:
        obs, _ = env.reset()
        terminated = False
        truncated = False
        step_count = 0
        current_episode_data = [] 
        
        print(f"\n--- Grabando Partida {successful_episodes + 1}/{num_episodes} ---")
        
        while not (terminated or truncated):
            env.render()
            
            action = None
            valid_key = False
            
            # Bucle de espera de tecla
            while not valid_key:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            print("⚠️ Cancelando grabación (Guardando lo que hay)...")
                            env.close()
                            if len(data) > 0:
                                save_data(data)
                            return
                        
                        # --- MAPEO DE TECLAS ARCADE ---
                        elif event.key == pygame.K_LEFT:
                            action = 0; valid_key = True
                        elif event.key == pygame.K_RIGHT:
                            action = 1; valid_key = True
                        elif event.key == pygame.K_UP:
                            action = 2; valid_key = True
                        elif event.key == pygame.K_DOWN:
                            action = 3; valid_key = True
                        elif event.key == pygame.K_a:
                            action = 4; valid_key = True # Coger
                        elif event.key == pygame.K_SPACE:
                            action = 5; valid_key = True # Abrir
            
            # Guardar paso
            current_episode_data.append({'obs': np.array(obs), 'action': int(action)})
            
            # Ejecutar acción
            obs, reward, terminated, truncated, _ = env.step(action)
            step_count += 1
            pygame.time.wait(80) # Velocidad cómoda

        # Criterio de éxito: Reward > 0
        if reward > 0 and terminated:
            print(f"✅ ¡Victoria en {step_count} pasos! Guardada.")
            data.extend(current_episode_data)
            successful_episodes += 1
        else:
            print("❌ Fallaste (Lava o Tiempo). Repitiendo partida...")

    env.close()
    save_data(data)

def save_data(data):
    filename = 'demonstrations_arcade.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"\n💾 ¡GRABACIÓN COMPLETADA! {len(data)} pasos guardados en '{filename}'")

if __name__ == "__main__":
    record_demonstrations(num_episodes=50)