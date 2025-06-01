#
# ISCTE-IUL, IAR, 2024/2025.
#
# Template to use SB3 to train a Thymio in Webots.
#

try:
    import sys
    import time
    import gymnasium as gym
    import numpy as np
    import math
    from stable_baselines3.common.callbacks import CheckpointCallback
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO
    from controller import Supervisor
    import random
    
except ImportError:
    sys.exit('Please make sure you have all dependencies installed.')


def random_orientation():
    angle = np.random.uniform(0, 2 * np.pi)
    return (0, 0, 1, angle)

def random_position(min_radius, max_radius, z):
    radius = np.random.uniform(min_radius, max_radius)
    angle = random_orientation()
    x = radius * np.cos(angle[3])
    y = radius * np.sin(angle[3])
    return (x, y, z)
    




#
# Structure of a class to create an OpenAI Gym in Webots.
#
class OpenAIGymEnvironment(Supervisor, gym.Env):
    def __init__(self, max_episode_steps = 500):
        super().__init__()
        self.spec = gym.envs.registration.EnvSpec(id='WebotsEnv-v0', entry_point='openai_gym:OpenAIGymEnvironment', max_episode_steps=max_episode_steps)
        self.timestep = int(self.getBasicTimeStep())

        # Fill in according to the action space of Thymio
        # See: https://www.gymlibrary.dev/api/spaces/
        self.action_space = gym.spaces.Box(
            low=np.array([-9.0,-9.0]), 
            high=np.array([9.0,9.0]), dtype=np.float32)

        # Fill in according to Thymio's sensors
        # See: https://www.gymlibrary.dev/api/spaces/
        self.observation_space = gym.spaces.Box(
            low=np.array([0,0,0,0,0,0,0]), 
            high=np.array([1,1,1,1,1,1,1]), dtype=np.float32)

        self.state = np.array([0,0,0,0,0,0,0])
        self.__n = 0
        
        self.posicao = [[0.0,0.0] for x in range(0,10)]
        self.posicao[3] = [1.0,2.0]
        
        self.grid_size = 0.1  # cena de percorrer o mapa
        self.visited_cells = set()
        
        self.override_canto = 0
        self.override_canto_conta = 0

        
        self.generate_boxes()
     
        
        self.sensores_frente = [self.getDevice(f'prox.horizontal.{i}') for i in range(5)]
        for sensor in self.sensores_frente:
            sensor.enable(self.timestep)
        
        self.sensores_baixo = [self.getDevice(f'prox.ground.{i}') for i in range(2)]
        for sensor in self.sensores_baixo:
            sensor.enable(self.timestep)


        
        self.left_motor = self.getDevice('motor.left')
        self.right_motor = self.getDevice('motor.right')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        

    def generate_boxes(self):
        N = 5
    
        for i in range(N):
            position = random_position(0.5, 0.5, 1)
            orientation = random_orientation()
            length = np.random.uniform(0.05, 0.2)
            width = np.random.uniform(0.05, 0.2)
            
            box_string = f"""
            DEF WHITE_BOX_{i} Solid {{
              translation {position[0]} {position[1]} {position[2]}
              rotation {orientation[0]} {orientation[1]} {orientation[2]} {orientation[3]}
              physics Physics {{
                density 1000.0
              }}
              children [
                Shape {{
                  appearance Appearance {{
                    material Material {{
                      diffuseColor 1 1 1
                    }}
                  }}
                  geometry Box {{
                    size {length} {width} 0.2  
                  }}
                }}
              ]
              boundingObject Box {{
                size {length} {width} 0.2  
              }}
            }}
            """
            self.getRoot().getField('children').importMFNodeFromString(-1, box_string)

    def del_boxes(self):
        """Remove all boxes from the scene"""
        for i in range(self.getRoot().getField('children').getCount()):
            node = self.getRoot().getField('children').getMFNode(i)
            if node.getTypeName() == "Solid" and node.getDef() and node.getDef().startswith("WHITE_BOX_"):
                self.getRoot().getField('children').removeMF(i)
                return self.del_boxes()  # Recursive call since indices change after removal
        return
    #
    # Reset the environment to an initial internal state, returning an initial observation and info.
    #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulationReset()
        self.simulationResetPhysics()
        super().step(self.timestep)

        for i in range(25):
            super().step(self.timestep)
        
        self.getSelf().getField('rotation').setSFRotation([0, 0, 1, np.random.uniform(0, 2 * np.pi)])
        self.getSelf().getField('translation').setSFVec3f([0, 0, 1])
        self.__n = 0
        
        #print(f"num cells visitadas: {len(self.visited_cells)}")
        #print(self.visited_cells)
        self.visited_cells = set()
        
        self.posicao = [[0.0,0.0] for x in range(0,10)]
        self.posicao[3] = [1.1,2.2]
        
        self.override_canto = 0
        self.override_canto_conta = 0


        # tenho que chamar isto outravez por causa do reset physics        
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

        
        for i in range(10):
            super().step(self.timestep)
            
        # caixas novas
        self.del_boxes()
        self.generate_boxes()
        
        # iteracoes para deixar a fisica acalmar um pouco
        for i in range(10):
            super().step(self.timestep)
        
        prox_values = [s.getValue() / 4096.0 for s in self.sensores_frente]
        ground_values = [s.getValue() / 4096.0 for s in self.sensores_baixo]

        self.state = np.array(prox_values + ground_values)

        return self.state.astype(np.float32), {}


    #
    # Run one timestep of the environment’s dynamics using the agent actions.
    #   
    def step(self, action):
        
        self.__n += 1
        
        # start by applying the action in the robot actuators
        
        #self.left_motor.setVelocity(float(action[0]))
        #self.right_motor.setVelocity(float(action[1]))

        # let the action to effect for a few timesteps
        for i in range(10):
            super().step(self.timestep)
            
        prox_values = [s.getValue() / 4096.0 for s in self.sensores_frente]
        ground_values = [s.getValue() / 4096.0 for s in self.sensores_baixo]
        z = self.getSelf().getField('translation').getSFVec3f()[2]
            
        # set the state that resulted from applying the action (consulting the robot sensors)
        self.state = np.array(prox_values + ground_values)
         
        
        x, y, _ = self.getSelf().getField('translation').getSFVec3f()
        cell_x = int(x / self.grid_size)
        cell_y = int(y / self.grid_size)
        cell = (cell_x, cell_y)
        
        self.posicao.pop(0)
        self.posicao.append([round(x,4),round(y,4)])
        parado = all(ok == self.posicao[0] for ok in self.posicao)
        
        if parado and not any(a > 0 for a in prox_values):
            reward = -2
            terminated = True
            truncated = False
            print("FIQUEI PRESO")
        else:
            """
            # compute the reward that results from applying the action in the current state
            #proximity_penalty = -np.sum(np.exp(-np.array(prox_values))) +10
            esta_parado = 0
            if parado:
                esta_parado = 3
            turn_away_bonus_prox = 0
            if any(a >= 0.8 for a in prox_values):
                if action[1] < action[0] or action[0] < action[1]:
                    turn_away_bonus_prox += 2
                proximity_penalty = -3 + turn_away_bonus_prox
            elif any(a >= 0.5 for a in prox_values):
                if action[1] < action[0] or action[0] < action[1]:
                    turn_away_bonus_prox += 2
                proximity_penalty = -2 + turn_away_bonus_prox
            elif any(a >= 0.2 for a in prox_values):
                if action[1] < action[0] or action[0] < action[1]:
                    turn_away_bonus_prox += 2
                proximity_penalty = -1 + turn_away_bonus_prox
            elif any(a > 0 for a in prox_values):
                if action[1] < action[0] or action[0] < action[1]:
                    turn_away_bonus_prox += 2
                proximity_penalty = -.5 + turn_away_bonus_prox
            else:
                proximity_penalty = 0
            
            left_drop = ground_values[0] < 0.1
            right_drop = ground_values[1] < 0.1
            
            is_near_drop = left_drop or right_drop
            ground_penalty = 5.0 if z<0.9 else 0.0
    
            turn_away_bonus_ground = 0
            linear_velocity = (action[0] + action[1]) / 2.0
            if is_near_drop:
                if left_drop and action[0] < action[1]:
                    turn_away_bonus_ground += 2.0
                if right_drop and action[1] < action[0]:
                    turn_away_bonus_ground += 2.0
                linear_velocity_reward = -abs(linear_velocity) + turn_away_bonus_ground
            else:
                linear_velocity_reward = linear_velocity * .5
                
            #linear_velocity = (action[0] + action[1]) / 2.0
            #angular_velocity = abs(action[0] - action[1])
            #linear_velocity_reward = -abs(linear_velocity) + angular_velocity + turn_away_bonus if is_near_drop else linear_velocity
            
            if cell not in self.visited_cells:
                exploration_bonus = 2
                self.visited_cells.add(cell)
            else:
                exploration_bonus = 0
            
    
            print(f"proximity_penalty: {proximity_penalty}\nground: {ground_penalty}\nvel: {linear_velocity_reward}\nparado: {esta_parado}\nexpoloracao: {exploration_bonus} ")
            reward = -3+linear_velocity_reward + proximity_penalty - ground_penalty - esta_parado + exploration_bonus
            print(f"reward: {reward}")
    
    
            # set termination and truncation flags (bools)
            terminated = z < 0.8 #is_near_drop#
            truncated = self.__n > self.spec.max_episode_steps
            """
            
            both_drop = ground_values[0] < 0.1 and ground_values[1] < 0.1
            #is_falling = z < 0.9 or (ground_values[0] < 0.05 or ground_values[1] < 0.05)
            left_drop = ground_values[0] < 0.1
            right_drop = ground_values[1] < 0.1
            reward = 0.0
            terminated = False
            truncated = False
            
            if self.override_canto_conta >0:
                self.override_canto_conta -= 1
                print("tou no canto")
                print(self.override_canto_conta)
                
                if self.override_canto == 1:
                    print("tou no canto 11111")
                    self.left_motor.setVelocity(3.0)
                    self.right_motor.setVelocity(-3.0)
                elif self.override_canto == 2:
                    print("tou no canto 22222")
                    self.left_motor.setVelocity(-3.0)
                    self.right_motor.setVelocity(3.0)
                elif self.override_canto == 3:
                    print("tou no canto 33333")
                    if self.override_canto_conta>15:
                        self.left_motor.setVelocity(-3.0)
                        self.right_motor.setVelocity(-3.0)
                        print("tou no canto 33333 escolha")
                    else:
                        self.override_canto = random.choice([1,2])
                        print(f"escolhi {self.override_canto}")
                
                return self.state.astype(np.float32), 0, False, False, {}
        
            # override
            if both_drop:
                self.left_motor.setVelocity(-3.0)
                self.right_motor.setVelocity(-3.0)
                reward -= 1.0
                self.override_canto_conta = 25
                self.override_canto = 3
            elif left_drop:
                # Turn right to recover
                self.left_motor.setVelocity(3.0)
                self.right_motor.setVelocity(-3.0)
                reward -= 1.0
                self.override_canto_conta = 10
                self.override_canto = 1
            elif right_drop:
                # Turn left to recover
                self.left_motor.setVelocity(-3.0)
                self.right_motor.setVelocity(3.0)
                reward -= 1.0
                self.override_canto_conta = 10
                self.override_canto = 2
            else:
                # Normal motion
                self.left_motor.setVelocity(float(action[0]))
                self.right_motor.setVelocity(float(action[1]))
        
            for _ in range(10):
                super().step(self.timestep)
        
            # Redetect after motion
            prox_values = [s.getValue() / 4096.0 for s in self.sensores_frente]
            ground_values = [s.getValue() / 4096.0 for s in self.sensores_baixo]
            self.state = np.array(prox_values + ground_values)
        
        
            if cell not in self.visited_cells:
                self.visited_cells.add(cell)
                reward += 2.0
        
            max_prox = max(prox_values)
            if max_prox > 0.8:
                reward -= 3.0
            elif max_prox > 0.5:
                reward -= 2.0
            elif max_prox > 0.2:
                reward -= 1.0
        
        # caiu
        if z<0.9:
            reward -= 10.0
            terminated = True
    
        # Reward forward motion
        linear_velocity = (action[0] + action[1]) / 2.0
        reward += max(0.0, 0.5 * linear_velocity)
    
        truncated = self.__n > self.spec.max_episode_steps
        print(f"reward: {reward}")

            
        return self.state.astype(np.float32), reward, terminated, truncated, {}
        
    
    # FUNCAO DE TESTES
    # para testar os rewards e isso, é mais facil do que tar a correr tudo
    def testes(self,steps):
        self.reset()
        print(f"{'Step':<6} {'Ground Left':<15} {'Ground Right':<15}")
        for i in range(steps):
            self.step([1.0, 1.0])  # Forward movement
            action = [1.0,2.0]
            ground_values = [s.getValue() / 4096.0 for s in self.sensores_baixo]
            prox_values = [s.getValue() / 4096.0 for s in self.sensores_frente]
            x, y, _ = self.getSelf().getField('translation').getSFVec3f()
            cell_x = int(x / self.grid_size)
            cell_y = int(y / self.grid_size)
            cell = (cell_x, cell_y)
            #proximity_penalty = np.sum(np.exp(-np.array(prox_values)))
            #teste = np.exp(-np.sum(prox_values)*3)*5
            #ground_penalty = np.sum(np.array(ground_values))
            #print(ground_values)
            
            # Penalização por estar perto de obstáculos
            reward = 0
            proximity_penalty = sum(min(p, 1.0) for p in prox_values) * -0.5
        
            # Recompensa por fugir de obstáculos
            turn_bonus = 0
            if any(p > 0 for p in prox_values):
                if abs(action[0] - action[1]) > 1:
                    turn_bonus = 1.5
                elif abs(action[0] - action[1]) > 0.5:
                    turn_bonus = 1
            
        
            # Penalização por estar perto de queda
            is_near_drop = any(g < 0.1 for g in ground_values)
            ground_penalty = -5.0 if is_near_drop else 0.0
        
            edge_avoidance_bonus = 0
            if is_near_drop:
                if abs(action[0] - action[1]) > 1 or (action[0] + action[1]) < 0:
                    edge_avoidance_bonus = 2.0
                elif abs(action[0] - action[1]) > 0.5:
                    edge_avoidance_bonus = 1.0
            
            # Recompensa por velocidade controlada
            linear_velocity = (action[0] + action[1]) / 2.0
            if not is_near_drop:
                velocity_reward = max(0, linear_velocity) * 0.3
            else:
                velocity_reward = -abs(linear_velocity)
        
            # Bónus por explorar célula nova
            exploration_bonus = 1.0 if cell not in self.visited_cells else 0.0
            if exploration_bonus > 0:
                self.visited_cells.add(cell)
            
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

def main():

    # Create the environment to train / test the robot
    env = OpenAIGymEnvironment()
    
    # testes:
    #env.testes(20000)
    
    # Code to train and save a model
    # For the PPO case, see how in Lab 7 code
    # For the RecurrentPPO case, consult its documentation
    checkpoint_callback = CheckpointCallback(save_freq=3000, save_path='modelos/', name_prefix='segundo_modelo_nao_finalizado_RPPO')

    # Code to load a model and run it
    # For the RecurrentPPO case, consult its documentation
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)#PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=300000, callback=checkpoint_callback)
    model.save("segundo_modelo_final_RPPO")
    
    
    # dar load ao modelo guardado
    """
    model = RecurrentPPO.load("nono_modelo_final")
    obs, _ = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    while True:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_starts = np.array([terminated or truncated])
        if terminated or truncated:
            obs, _ = env.reset()
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
    """
if __name__ == '__main__':
    main()
