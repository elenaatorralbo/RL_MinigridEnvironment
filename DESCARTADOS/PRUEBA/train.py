import numpy as np
import random
import pickle
from env import SequentialRoomsEnv

class SimpleRLAgent:
    def __init__(self, action_space_n):
        self.action_space_n = action_space_n
        self.q_table = {} 
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999 

    def get_state_key(self, obs):
        return tuple(obs)

    def get_action(self, obs):
        state = self.get_state_key(obs)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_n - 1)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space_n)
        return np.argmax(self.q_table[state])

    def update(self, obs, action, reward, next_obs, terminated):
        state = self.get_state_key(obs)
        next_state = self.get_state_key(next_obs)
        if state not in self.q_table: self.q_table[state] = np.zeros(self.action_space_n)
        if next_state not in self.q_table: self.q_table[next_state] = np.zeros(self.action_space_n)

        current_q = self.q_table[state][action]
        next_q_values = self.q_table[next_state]
        policy_probs = np.ones(self.action_space_n) * (self.epsilon / self.action_space_n)
        best_next_action = np.argmax(next_q_values)
        policy_probs[best_next_action] += (1.0 - self.epsilon)
        expected_next_q = np.sum(policy_probs * next_q_values)
        
        target = reward + (self.gamma * expected_next_q * (1 - int(terminated)))
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"[Sistema] Modelo guardado en: {filename}")

def train():
    # --- CAMBIO 1: REDUCIR MAX_STEPS ---
    # Bajamos de 2000 a 300. Si no lo resuelve en 300 pasos (para 1 o 2 habitaciones),
    # es que no sabe hacerlo. Cortamos para reiniciar rápido.
    env = SequentialRoomsEnv(render_mode=None, max_steps=300) 
    agent = SimpleRLAgent(env.action_space.n)
    
    # Intentar cargar conocimiento previo si existe
    try:
        with open("q_table_level_19.pkl", "rb") as f:
            agent.q_table = pickle.load(f)
        print("Cerebro previo cargado.")
    except:
        pass

    start_room = 19
    episodes_per_check = 50 
    win_history = []
    total_episodes = 50000 
    
    print(f"--- Iniciando Entrenamiento VELOZ (Nivel: {start_room}) ---")

    for episode in range(total_episodes):
        env.set_start_room(start_room)
        
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        
        while not (terminated or truncated):
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            # Penalización por paso
            if reward == 0:
                reward = -0.01

            agent.update(obs, action, reward, next_obs, terminated)
            
            obs = next_obs
            total_reward += reward
            
            # --- CAMBIO 2: MERCY KILL (Muerte por Piedad) ---
            # Si el agente lo está haciendo fatal (ej: -5 puntos acumulados),
            # cortamos el episodio ya. Esto acelera el entrenamiento x10.
            if total_reward < -5.0:
                terminated = True # Forzamos fin
                reward = -10 # Castigo extra por ser lento/malo

        agent.decay_epsilon()

        is_victory = total_reward > 40 
        win_history.append(1 if is_victory else 0)
        
        if len(win_history) >= episodes_per_check:
            recent_wins = win_history[-episodes_per_check:]
            win_rate = sum(recent_wins) / len(recent_wins)
            
            # Imprimimos menos a menudo para no ensuciar la consola, o igual.
            print(f"Ep {episode+1} | Room {start_room} | Win: {win_rate:.2f} | Eps: {agent.epsilon:.2f}")
            
            win_history = [] 

            if win_rate >= 0.85 and agent.epsilon < 0.15:
                agent.save(f"q_table_level_{start_room}.pkl")
                
                if start_room > 0:
                    start_room -= 1
                    print(f"\n>>> MAESTRÍA DEMOSTRADA. Avanzando a Habitación {start_room}\n")
                    agent.epsilon = max(agent.epsilon, 0.4) 
                else:
                    print("\n>>> ¡JUEGO COMPLETADO! El agente es un experto real.")
                    agent.save("q_table_MASTER.pkl")
                    break
    
    agent.save("q_table_final.pkl")
    env.close()

if __name__ == "__main__":
    train()