import gymnasium as gym
import torch
import numpy as np
import imageio
import os
import sys
from minigrid.wrappers import ImgObsWrapper
from sb3_contrib import RecurrentPPO
from environment import SurvivalCorridorEnv

# --- CONFIGURACIÓN ---
VELOCIDAD_VIDEO = 40 
PASOS_MAXIMOS_VIDEO = 2000 # <-- Cortamos aquí para no esperar eternamente
# ---------------------

def main():
    print(f"🎥 Preparando grabación (Máx {PASOS_MAXIMOS_VIDEO} pasos por video)...")
    print("💡 TRUCO: Si te aburres, pulsa Ctrl+C y el video SE GUARDARÁ igual.")
    
    env = SurvivalCorridorEnv(render_mode="rgb_array", num_rooms=25, agent_view_size=9)
    env = ImgObsWrapper(env)

    model_path = "mejores_modelos/best_model.zip"
    if not os.path.exists(model_path):
        print(f"❌ Error: No se encuentra {model_path}")
        return

    try:
        model = RecurrentPPO.load(model_path, env=env)
        print("✅ ¡Modelo cargado!")
    except Exception as e:
        print(f"❌ Error cargando el modelo: {e}")
        return

    for episodio in range(1, 2): # Grabamos solo 1 episodio para probar
        obs, _ = env.reset()
        
        lstm_states = None 
        episode_starts = np.ones((1,), dtype=bool)
        
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        frames = [] 
        
        print(f"\n🔴 Grabando Episodio {episodio}...")
        
        try:
            while not terminated and not truncated:
                frame = env.render()
                frames.append(frame)

                action, lstm_states = model.predict(
                    obs, 
                    state=lstm_states, 
                    episode_start=episode_starts,
                    deterministic=False 
                )
                
                episode_starts[0] = False
                
                if isinstance(action, np.ndarray):
                    action = int(action.item())
                
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                # --- PARCHE EN CALIENTE ---
                if hasattr(env.unwrapped, 'agent_pos') and isinstance(env.unwrapped.agent_pos, np.ndarray):
                    env.unwrapped.agent_pos = tuple(env.unwrapped.agent_pos)
                if hasattr(env.unwrapped, 'agent_dir') and isinstance(env.unwrapped.agent_dir, (np.ndarray, np.integer)):
                    env.unwrapped.agent_dir = int(env.unwrapped.agent_dir)
                # --------------------------

                if steps % 50 == 0:
                    print(f"   ... Paso {steps}")

                # LÍMITE DE SEGURIDAD
                if steps >= PASOS_MAXIMOS_VIDEO:
                    print(f"⚠️ Límite de video alcanzado ({PASOS_MAXIMOS_VIDEO} pasos). Cortando grabación.")
                    break

        except KeyboardInterrupt:
            print("\n🛑 ¡Interrupción de usuario detectada!")
            print("⏳ Guardando lo grabado hasta ahora (no cierres)...")

        # Guardar video (pase lo que pase)
        if len(frames) > 0:
            nombre_video = f"video_episodio_{episodio}.mp4"
            print(f"💾 Guardando {nombre_video} ({len(frames)} frames)...")
            imageio.mimsave(nombre_video, frames, fps=VELOCIDAD_VIDEO) 
            print(f"✅ ¡Video guardado!")
        else:
            print("❌ No se grabaron frames.")

    env.close()

if __name__ == "__main__":
    main()