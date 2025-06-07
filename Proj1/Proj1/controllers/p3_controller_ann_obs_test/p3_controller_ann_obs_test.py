import numpy as np
from controller import Supervisor
import os
import time
from collections import deque
import math

TIME_STEP = 5
EVALUATION_TIME = 100
RANGE = 9

INPUT = 5
HIDDEN1 = 11
HIDDEN2 = 7 
OUTPUT = 2

GROUND_SENSOR_THRESHOLD = 200
PROXIMITY_SENSOR_THRESHOLD_FOR_COLLISION = 2000

BEST_WEIGHTS_FILE = "best_weights_obstacles.npy"

def random_orientation():
    angle = np.random.uniform(0, 2 * np.pi)
    return [0, 0, 1, angle]
    
def random_position(min_radius, max_radius, z):
    r = np.sqrt(np.random.uniform(min_radius**2, max_radius**2))  
    theta = np.random.uniform(0, 2 * np.pi)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return (x, y, z)
    
    
class SimpleANN:
    def __init__(self, weights):
        # Calculate weight matrix indices for two hidden layers
        idx_w1 = HIDDEN1 * (INPUT + 1)
        idx_w2 = idx_w1 + HIDDEN2 * (HIDDEN1 + 1)
        
        # Split weights into three matrices
        self.w1 = weights[:idx_w1].reshape(HIDDEN1, INPUT + 1)                    # Input to Hidden1
        self.w2 = weights[idx_w1:idx_w2].reshape(HIDDEN2, HIDDEN1 + 1)           # Hidden1 to Hidden2
        self.w3 = weights[idx_w2:].reshape(OUTPUT, HIDDEN2 + 1)                  # Hidden2 to Output
    
    def forward(self, inputs):
        # Input layer to first hidden layer
        h1 = np.tanh(np.dot(self.w1, np.append(inputs, 1.0)))
        
        # First hidden layer to second hidden layer  
        h2 = np.tanh(np.dot(self.w2, np.append(h1, 1.0)))
        
        # Second hidden layer to output layer - MATCH TRAINING EXACTLY
        return np.tanh(np.dot(self.w3, np.append(h2, 1.0))) * 9

class VisualizeSolution:
    def __init__(self):
        self.supervisor = Supervisor()
        self.robot_node = self.supervisor.getFromDef("ROBOT")
        self.root = self.supervisor.getRoot()

        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")
        self.basic_timestep = int(self.supervisor.getBasicTimeStep())
        self.timestep = self.basic_timestep * TIME_STEP

        self.left_motor = self.supervisor.getDevice('motor.left')
        self.right_motor = self.supervisor.getDevice('motor.right')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.max_motor_velocity = self.left_motor.getMaxVelocity()

        # Initialize sensors exactly like training
        self.prox_sensors = [self.supervisor.getDevice(f'prox.horizontal.{i}') for i in range(7)]
        self.ground_sensors = [self.supervisor.getDevice(f'prox.ground.{i}') for i in range(2)]

        for sensor in self.prox_sensors + self.ground_sensors:
            sensor.enable(self.timestep)

    def get_sensor_data(self):
        """Get sensor data exactly like training code"""
        prox = [self.prox_sensors[i].getValue() for i in [0, 2, 4]]
        prox_norm = [(v / 4230.0) * 2.0 - 1.0 for v in prox]

        ground_norm = [(s.getValue() / 750.0) for s in self.ground_sensors]

        return np.array(prox_norm + ground_norm)
        
    def runStepLogic(self, ann_model):
        # Use the same sensor data processing as training
        inputs = self.get_sensor_data()
        velocities = ann_model.forward(inputs)

        left_vel = np.clip(velocities[0], -self.max_motor_velocity, self.max_motor_velocity)
        right_vel = np.clip(velocities[1], -self.max_motor_velocity, self.max_motor_velocity)
        
        self.left_motor.setVelocity(left_vel)
        self.right_motor.setVelocity(right_vel)

        return True
        
    def generate_boxes(self):
        # Use same parameters as training (8 boxes, same radius range)
        for i in range(8):
            position = random_position(0.7, 1.6, 1)  # Match training parameters
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
            self.root.getField('children').importMFNodeFromString(-1, box_string)

    def del_boxes(self):
        """Remove all boxes from the scene"""
        children_field = self.root.getField('children')
        for i in range(children_field.getCount() - 1, -1, -1):
            node = children_field.getMFNode(i)
            if node and node.getTypeName() == "Solid" and node.getDef() and node.getDef().startswith("WHITE_BOX_"):
                children_field.removeMF(i)

    def reset(self):
        self.robot_node.resetPhysics()
        # Match training reset position exactly
        self.translation_field.setSFVec3f([0, 0, 0.02])
        self.rotation_field.setSFRotation(random_orientation())

        self.supervisor.step(self.basic_timestep)

        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        
        # Generate boxes
        self.generate_boxes()
        print("Generated obstacle boxes")

    def _load_weights_from_file(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weights file not found: {filepath}")
        loaded_weights = np.load(filepath)
        print(f"Loaded weights shape: {loaded_weights.shape}")
        return loaded_weights

    def run_visualization(self, weights_file=None):
        # Load weights
        if weights_file:
            weights_to_run = self._load_weights_from_file(weights_file)
            print(f"Loaded weights from: {weights_file}")
        else:
            weights_to_run = self._load_weights_from_file(BEST_WEIGHTS_FILE)
            print(f"Loaded weights from: {BEST_WEIGHTS_FILE}")

        # Set simulation to real-time mode
        self.supervisor.simulationSetMode(self.supervisor.SIMULATION_MODE_REAL_TIME)
        self.supervisor.step(self.basic_timestep)

        # Reset robot position
        self.reset()

        # Create neural network with loaded weights
        ann_instance = SimpleANN(weights_to_run)
        print("Neural network initialized. Starting robot...")

        start_time = self.supervisor.getTime()

        # Main simulation loop
        while self.supervisor.step(self.timestep) != -1:
            current_sim_time = self.supervisor.getTime()
            
            # Stop after evaluation time
            if current_sim_time - start_time >= EVALUATION_TIME:
                print(f"Simulation completed after {EVALUATION_TIME} seconds")
                break

            # Run one step of robot control
            self.runStepLogic(ann_instance)

        print("Robot simulation finished.")
        print('-'*50)


if __name__ == "__main__":
    visualizer = VisualizeSolution()
    visualizer.run_visualization()