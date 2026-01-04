# visualize.py
from minigrid.manual_control import ManualControl
from environment import SurvivalCorridorEnv  # Importamos tu clase

def main():
    # Es CRUCIAL poner render_mode="human" para que se abra la ventana
    env = SurvivalCorridorEnv(render_mode="human")
    
    # Esta herramienta de MiniGrid abre la ventana y captura tu teclado
    manual_control = ManualControl(env)
    manual_control.start()

if __name__ == "__main__":
    main()