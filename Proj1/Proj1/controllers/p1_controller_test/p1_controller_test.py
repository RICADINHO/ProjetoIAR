import numpy as np
from controller import Supervisor
import random

# Simulation parameters
TIME_STEP = 5

def random_orientation():
    angle = np.random.uniform(0, 2*np.pi)
    return [0, 0, 1, angle]

class SimpleRun:
    def __init__(self):
        # Supervisor setup
        self.supervisor = Supervisor()
        self.robot = self.supervisor.getSelf()
        self.robot_node = self.supervisor.getFromDef("ROBOT")
        
        self.basic_timestep = int(self.supervisor.getBasicTimeStep())
        self.timestep = self.basic_timestep * TIME_STEP
        
        # Motors and sensors
        self.left_motor = self.supervisor.getDevice('motor.left')
        self.right_motor = self.supervisor.getDevice('motor.right')
        self.ground_sensors = [self.supervisor.getDevice(f'prox.ground.{i}') for i in range(2)]

        for motor in [self.left_motor, self.right_motor]:
            motor.setPosition(float('inf'))

        for sensor in self.ground_sensors:
            sensor.enable(self.timestep)

        # Load best weights
        try:
            self.weights = np.load("best_weights_direct.npy")
            print(f"Loaded weights: {self.weights}")
        except FileNotFoundError:
            print("Error: best_weights_direct.npy not found!")
            exit(1)

    def run(self):
        """Run the robot continuously with best weights"""
        print("=== Running Robot with Best Weights ===")
        
        initial_pos = [0, 0, 0.02]
        random_rotation = random_orientation()
        
        self.robot.getField('translation').setSFVec3f(initial_pos)
        self.robot.getField('rotation').setSFRotation(random_rotation)
        
        
        while self.supervisor.step(self.timestep) != -1:
            # Ground sensor processing
            ground_left = (self.ground_sensors[0].getValue() / 750)
            ground_right = (self.ground_sensors[1].getValue() / 750)
            
            # Direct motor control using best weights
            left_speed = (ground_left * self.weights[0] + 
                         ground_right * self.weights[1] + 
                         self.weights[2])
            right_speed = (ground_left * self.weights[3] + 
                          ground_right * self.weights[4] + 
                          self.weights[5])
            
            # Clip speeds and set motors
            left_speed = max(min(left_speed, 9), -9)
            right_speed = max(min(right_speed, 9), -9)
            
            self.left_motor.setVelocity(left_speed)
            self.right_motor.setVelocity(right_speed)

if __name__ == "__main__":
    runner = SimpleRun()
    runner.run()
