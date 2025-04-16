import numpy as np
from controller import Supervisor
import random
import math
import numpy as np

# Simulation parameters
TIME_STEP = 5
POPULATION_SIZE = 10
PARENTS_KEEP = 3
INPUT = 5
HIDDEN = 4
OUTPUT = 2
GENOME_SIZE = (1+INPUT)*HIDDEN  + (HIDDEN+1)*OUTPUT
GENERATIONS = 300
MUTATION_RATE = 0.2
MUTATION_SIZE = 0.05
EVALUATION_TIME = 300  # Simulated seconds per individual
RANGE = 5


def random_orientation():
    angle = np.random.uniform(0, 2 * np.pi)
    return (0, 0, 1, angle)

def random_position(min_radius, max_radius, z):
    radius = np.random.uniform(min_radius, max_radius)
    angle = random_orientation()
    x = radius * np.cos(angle[3])
    y = radius * np.sin(angle[3])
    return (x, y, z)

class Evolution:
    def __init__(self):
        self.evaluation_start_time = 0
        self.collision = False

        # Supervisor to reset robot position
        self.supervisor = Supervisor()
        self.robot = self.supervisor.getSelf()
   
        self.robot_node = self.supervisor.getFromDef("ROBOT") 
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")

        self.timestep = int(self.supervisor.getBasicTimeStep()*TIME_STEP)
        self.left_motor = self.supervisor.getDevice('motor.left')
        self.right_motor = self.supervisor.getDevice('motor.right')

        self.__ir_0 = self.supervisor.getDevice('prox.horizontal.0')
        self.__ir_1 = self.supervisor.getDevice('prox.horizontal.1')
        self.__ir_2 = self.supervisor.getDevice('prox.horizontal.2')
        self.__ir_3 = self.supervisor.getDevice('prox.horizontal.3')
        self.__ir_4 = self.supervisor.getDevice('prox.horizontal.4')
        self.__ir_5 = self.supervisor.getDevice('prox.horizontal.5')
        self.__ir_6 = self.supervisor.getDevice('prox.horizontal.6')
        self.__ir_7 = self.supervisor.getDevice('prox.ground.0')
        self.__ir_8 = self.supervisor.getDevice('prox.ground.1')

        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        self.__ir_0.enable(self.timestep)
        self.__ir_1.enable(self.timestep)
        self.__ir_2.enable(self.timestep)
        self.__ir_3.enable(self.timestep)
        self.__ir_4.enable(self.timestep)
        self.__ir_5.enable(self.timestep)
        self.__ir_6.enable(self.timestep)
        self.__ir_7.enable(self.timestep)
        self.__ir_8.enable(self.timestep)

        self.sensors = [self.__ir_0,self.__ir_2,self.__ir_4]
        self.ground_sensors = [self.supervisor.getDevice(f'prox.ground.{i}') for i in range(2)]

        self.__n = 0
        self.prev_position = self.supervisor.getSelf().getPosition()
        
        
   

    def reset(self, seed=None, options=None):
        
        random_rotation = [0, 0, 1, np.random.uniform(0, 2 * np.pi)]
        self.supervisor.getFromDef('ROBOT').getField('rotation').setSFRotation(random_rotation)
        self.supervisor.getFromDef('ROBOT').getField('translation').setSFVec3f([0, 0, 0])
        
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        

    def runStep(self, weights):
        
        self.collision = bool(
                self.__n > 10 and
                (self.__ir_0.getValue()>4300 or 
                self.__ir_1.getValue()>4300 or
                self.__ir_2.getValue()>4300 or
                self.__ir_3.getValue()>4300 or
                self.__ir_4.getValue()>4300 or
                self.__ir_5.getValue()>4300 or
                self.__ir_6.getValue()>4300)
            )
        
        ground_sensor_left = (self.ground_sensors[0].getValue()/1023 - .6)/.2>.3
        ground_sensor_right = (self.ground_sensors[1].getValue()/1023 - .6)/.2>.3

        left_speed =  ground_sensor_left * weights[0] + ground_sensor_right * weights[1] + weights[2]
        right_speed = ground_sensor_left * weights[3] + ground_sensor_right * weights[4] + weights[5]
        
        self.left_motor.setVelocity(max(min(left_speed, 9), -9))
        self.right_motor.setVelocity(max(min(right_speed, 9), -9))

        self.supervisor.step(self.timestep)



   
    def run(self):
        self.evaluation_start_time = self.supervisor.getTime()
        weights = [1,1,1,1,1,2]
        while self.supervisor.getTime() - self.evaluation_start_time < EVALUATION_TIME and not self.collision:
            self.runStep(weights)


# Main evolutionary loop
def main():

    # Run the evolutionary algorithm
    controller = Evolution()
    controller.run()

if __name__ == "__main__":
    main()

