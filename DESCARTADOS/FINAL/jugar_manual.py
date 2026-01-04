import pygame
import sys
from environment import SurvivalCorridorEnv

def main():
    # CONFIGURACIÓN
    # Pon start_room=24 para probar solo el final
    # Pon start_room=20 para probar el tramo difícil de lava
    START_ROOM = 24 
    
    print(f"Iniciando modo manual en la habitación {START_ROOM}...")
    print("CONTROLES:")
    print("  [FLECHAS]: Moverse (Derecha, Abajo, Izquierda, Arriba)")
    print("  [ESC]: Salir")
    print("----------------------------------------------------")

    # Inicializamos el entorno en modo 'human' para que use pygame
    env = SurvivalCorridorEnv(
        render_mode="human", 
        num_rooms=25, 
        room_size=7, 
        agent_view_size=9, 
        agent_start_room=START_ROOM
    )

    obs, _ = env.reset()
    env.render() # Primer renderizado

    running = True
    total_reward = 0
    step_count = 0

    while running:
        # Procesar eventos de teclado de Pygame
        # (MiniGrid usa su propia ventana, pero necesitamos capturar las teclas)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                
                action = None
                
                # TUS CONTROLES PERSONALIZADOS (Directos)
                # 0: Der, 1: Abajo, 2: Izq, 3: Arriba
                if event.key == pygame.K_RIGHT:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_UP:
                    action = 3
                
                # Si se pulsó una tecla válida, damos el paso
                if action is not None:
                    obs, reward, terminated, truncated, info = env.step(action)
                    step_count += 1
                    total_reward += reward
                    
                    print(f"Paso: {step_count} | Acción: {action} | Reward: {reward:.2f} | Total: {total_reward:.2f}")
                    
                    env.render()

                    if terminated or truncated:
                        print("\n" + "="*30)
                        if reward >= 50:
                            print("¡VICTORIA! META ALCANZADA 🎉")
                        else:
                            print("FIN DEL EPISODIO (Muerte o Tiempo)")
                        print(f"Reward Final: {total_reward:.2f}")
                        print("="*30 + "\n")
                        
                        # Reiniciar
                        obs, _ = env.reset()
                        total_reward = 0
                        step_count = 0
                        env.render()

    env.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()