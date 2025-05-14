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
    def __init__(self, max_episode_steps = 200):
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
        #self.simulationReset()
        #self.simulationResetPhysics()
        super().step(self.timestep)
        
        
        self.getSelf().getField('rotation').setSFRotation([0, 0, 1, np.random.uniform(0, 2 * np.pi)])
        self.getSelf().getField('translation').setSFVec3f([0, 0, 1])
        self.__n = 0
        
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

        
        for i in range(10):
            super().step(self.timestep)
            
        # caixas novas
        self.del_boxes()
        self.generate_boxes()
        
        for i in range(10):
            super().step(self.timestep)
        
        prox_values = [s.getValue() / 4096.0 for s in self.sensores_frente]
        ground_values = [s.getValue() / 4096.0 for s in self.sensores_baixo]

        # you may need to iterate a few times to let physics stabilize


        self.state = np.array(prox_values + ground_values)

        return self.state.astype(np.float32), {}


    #
    # Run one timestep of the environment’s dynamics using the agent actions.
    #   
    def step(self, action):

        self.__n += 1

        # start by applying the action in the robot actuators
        
        self.left_motor.setVelocity(float(action[0]))
        self.right_motor.setVelocity(float(action[1]))

        # let the action to effect for a few timesteps
        for i in range(10):
            super().step(self.timestep)
        
        prox_values = [s.getValue() / 4096.0 for s in self.sensores_frente]
        ground_values = [s.getValue() / 4096.0 for s in self.sensores_baixo]

        # set the state that resulted from applying the action (consulting the robot sensors)
        self.state = np.array(prox_values + ground_values)

        # compute the reward that results from applying the action in the current state
        proximity_penalty = np.sum(np.exp(-np.array(prox_values)))  # closer = lower
        ground_penalty = np.sum(np.array(ground_values) < 0.1)  # near edge or drop
        linear_velocity_reward = np.mean(action)  # prefer moving forward
        teste = -np.exp(-np.sum(prox_values)*3)*5 + 3
        print(f"proximity_penalty: {proximity_penalty}\nground: {ground_penalty}\nvel: {linear_velocity_reward}")
        reward = linear_velocity_reward - proximity_penalty - ground_penalty
        print(f"reward: {reward}")

        # set termination and truncation flags (bools)
        terminated = ground_penalty > 0
        truncated = self.__n > self.spec.max_episode_steps

        return self.state.astype(np.float32), reward, terminated, truncated, {}
    
    def testes(self,steps):
        self.reset()
        print(f"{'Step':<6} {'Ground Left':<15} {'Ground Right':<15}")
        for i in range(steps):
            self.step([1.0, 1.0])  # Forward movement
    
            ground_values = [s.getValue() / 4096.0 for s in self.sensores_baixo]
            prox_values = [s.getValue() / 4096.0 for s in self.sensores_frente]
            proximity_penalty = np.sum(np.exp(-np.array(prox_values)))
            teste = np.exp(-np.sum(prox_values)*3)*5
            ground_penalty = np.sum(np.array(ground_values))
            #print(f"{i:<6} {ground_values[0]:<15.4f} {ground_values[1]:<15.4f}")
            #print(proximity_penalty)
            #print(proximity_penalty)
            #print(ground_penalty)
            #print(np.sum(np.array(ground_values) < 0.1))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

def main():

    # Create the environment to train / test the robot
    env = OpenAIGymEnvironment()
    
    #env.testes(500)
    # Code to train and save a model
    # For the PPO case, see how in Lab 7 code
    # For the RecurrentPPO case, consult its documentation
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='modelos/', name_prefix='terceiro_modelo_nao_finalizado')

    # Code to load a model and run it
    # For the RecurrentPPO case, consult its documentation
    model = PPO("MlpPolicy", env, verbose=1 )
    model.learn(total_timesteps=100000, callback=checkpoint_callback)
    model.save("terceiro_modelo_final")
    
    
        # dar load ao modelo
    """model = PPO.load("segundo_modelo_final")
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
