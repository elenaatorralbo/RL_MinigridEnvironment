import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os

# =========================================================
# NUEVO ENFOQUE: SIMPLIFICACIÓN TOTAL
# =========================================================

if __name__ == "__main__":
    # 1. EL ENTORNO
    # Usamos uno oficial pequeño para garantizar que tiene solución.
    # 6x6 es perfecto: ni muy tonto ni muy imposible.
    env_id = "MiniGrid-DoorKey-6x6-v0"

    # 2. LA CLAVE DEL ÉXITO: FlatObsWrapper
    # Esto aplana la visión. El agente no ve una imagen 3D, ve una lista de números.
    # Es mucho más fácil encontrar patrones aquí.
    wrapper_class = FlatObsWrapper

    # 3. CREAR ENTORNOS PARALELOS
    # 8 agentes entrenando a la vez
    vec_env = make_vec_env(
        env_id,
        n_envs=8,
        wrapper_class=wrapper_class
    )

    print(f"--- Entrenando en {env_id} con VECTORES (No Imágenes) ---")

    # 4. EL MODELO: MlpPolicy
    # Cambiamos 'CnnPolicy' por 'MlpPolicy'.
    # Al no haber imágenes, usamos redes densas simples. Aprenden volando.
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,  # Un poco de curiosidad
        gamma=0.99,
        device="auto"
    )

    # 5. ENTRENAMIENTO
    # Con este enfoque, 1 millón de pasos se hacen en minutos, no horas.
    total_timesteps = 500_000
    model.learn(total_timesteps=total_timesteps)

    # Guardar
    save_path = "PPO_DoorKey_Flat"
    model.save(save_path)
    print("--- Entrenamiento finalizado ---")

    # =========================================================
    # VISUALIZACIÓN
    # =========================================================
    print("--- Probando el agente ---")

    # Creamos entorno de test con el mismo wrapper
    env = gym.make(env_id, render_mode="human")
    env = FlatObsWrapper(env)

    model = PPO.load(save_path)

    obs, _ = env.reset()

    while True:
        # Predecir acción
        action, _ = model.predict(obs, deterministic=True)

        # Ejecutar
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            print(f"Recompensa final: {reward:.2f}")
            if reward > 0:
                print(">>> ¡ÉXITO! HA LLEGADO A LA META <<<")
            obs, _ = env.reset()