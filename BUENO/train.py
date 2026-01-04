import gymnasium as gym
import minigrid
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium import spaces, ObservationWrapper
import os

# =============================================================================
# 1. WRAPPER DE IMAGEN (Ojos de la IA)
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
# 2. CONFIGURACIÓN DEL CURRICULUM
# =============================================================================
def run_final_curriculum():
    
    # Definimos las 3 Fases exactas
    stages = [
        # --- FASE 1: INICIO ---
        {
            "name": "1_Nivel_N2",
            "id": "MiniGrid-MultiRoom-N2-S4-v0",
            "steps": 200_000,
            "kwargs": {} 
        },
        
        # --- FASE 2: EL SALTO (N4 - Tamaño 5) ---
        {
            "name": "2_Nivel_N4_Size5",
            "id": "MiniGrid-MultiRoom-N4-S5-v0",
            "steps": 500_000,
            "kwargs": {} 
        },
        
        # --- FASE 3: LA RESISTENCIA (N4 - Tamaño 8) ---
        {
            "name": "3_Nivel_N4_Size8",
            "id": "MiniGrid-MultiRoom-N6-v0",
            "steps": 700_000,
            "kwargs": {"minNumRooms": 6, "maxNumRooms": 6, "maxRoomSize": 8}
        }
    ]

    # ### NUEVO: Definimos carpeta para logs
    log_dir = "./tensorboard_logs/"

    print("🚀 INICIANDO SCRIPT MAESTRO CON TENSORBOARD 🚀")
    print(f"   Logs se guardarán en: {log_dir}")
    print("================================================")
    
    model = None
    
    for i, stage in enumerate(stages):
        print(f"\n>>> [FASE {i+1}/{len(stages)}] Ejecutando: {stage['name']}")
        print(f"    Entorno: {stage['id']}")
        print(f"    Pasos: {stage['steps']}")

        # 1. Crear el entorno
        env = gym.make(stage["id"], render_mode=None, **stage["kwargs"])
        env = ImgObsWrapper(env)

        # 2. Gestionar el Modelo
        if model is None:
            print("    🧠 Creando cerebro desde CERO (PPO)...")
            model = PPO(
                "MlpPolicy", 
                env, 
                verbose=1, 
                learning_rate=0.0003,
                ent_coef=0.01,
                tensorboard_log=log_dir  # ### NUEVO: Activamos TensorBoard aquí
            )
        else:
            print("    🧠 Transfiriendo conocimientos de la fase anterior...")
            model.set_env(env)

        # 3. Checkpoints
        checkpoint_callback = CheckpointCallback(
            save_freq=100000,
            save_path=f"./checkpoints/{stage['name']}/",
            name_prefix=stage['name']
        )

        # 4. ¡ENTRENAR!
        model.learn(
            total_timesteps=stage['steps'], 
            callback=checkpoint_callback,
            reset_num_timesteps=True,
            tb_log_name=stage['name'] # ### NUEVO: Nombre de la curva en la gráfica
        )

        # 5. Guardar modelo final de la fase
        save_path = f"{stage['name']}_FINAL"
        model.save(save_path)
        print(f"✅ FASE {i+1} COMPLETADA. Modelo guardado en: {save_path}.zip")
        
        env.close()

    print("\n🏆 ¡TODO EL ENTRENAMIENTO HA FINALIZADO! 🏆")

if __name__ == "__main__":
    run_final_curriculum()