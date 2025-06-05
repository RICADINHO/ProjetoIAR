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

BEST_WEIGHTS_FILE = "best_weights_obstacules.npy"

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
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, inputs):
        # Input layer to first hidden layer
        inputs_with_bias = np.append(inputs, 1.0)
        hidden1_inputs = np.dot(self.w1, inputs_with_bias)
        hidden1_outputs = self.tanh(hidden1_inputs)
        
        # First hidden layer to second hidden layer
        hidden1_with_bias = np.append(hidden1_outputs, 1.0)
        hidden2_inputs = np.dot(self.w2, hidden1_with_bias)
        hidden2_outputs = self.tanh(hidden2_inputs)
        
        # Second hidden layer to output layer
        hidden2_with_bias = np.append(hidden2_outputs, 1.0)
        final_inputs = np.dot(self.w3, hidden2_with_bias)
        final_outputs = self.tanh(final_inputs)
        
        # Scale outputs
        motor_outputs = final_outputs * RANGE
        return motor_outputs

class VisualizeSolution:
    def __init__(self):
        self.supervisor = Supervisor()
        self.robot_node = self.supervisor.getFromDef("ROBOT")
        self.root = self.supervisor.getRoot()  # Add root node reference

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

        self.ground_sensors = []
        for i in range(2):
            sensor_name = f'prox.ground.{i}'
            sensor = self.supervisor.getDevice(sensor_name)
            sensor.enable(self.timestep)
            self.ground_sensors.append(sensor)

        self.ann_proximity_sensors = []
        self.collision_proximity_sensors = []

        ann_prox_indices = [0, 2, 4]

        for i in range(7):
            sensor_name = f'prox.horizontal.{i}'
            sensor = self.supervisor.getDevice(sensor_name)
            sensor.enable(self.timestep)
            self.collision_proximity_sensors.append(sensor)
            if i in ann_prox_indices:
                self.ann_proximity_sensors.append(sensor)

        
    def runStepLogic(self, ann_model):
        raw_ground_sensor_values = [gs.getValue() for gs in self.ground_sensors]
        gs_max_value = 1023.0
        normalized_ground_inputs = [(val / gs_max_value) * 2.0 - 1.0 for val in raw_ground_sensor_values]

        raw_ann_proximity_values = [ps.getValue() for ps in self.ann_proximity_sensors]
        prox_max_value = 4230.0
        normalized_ann_proximity_inputs = [(val / prox_max_value) * 2.0 - 1 for val in raw_ann_proximity_values]

        ann_inputs = np.concatenate((np.array(normalized_ground_inputs), np.array(normalized_ann_proximity_inputs)))

        motor_speeds = ann_model.forward(ann_inputs)
        left_speed, right_speed = float(motor_speeds[0]), float(motor_speeds[1])

        left_speed = np.clip(left_speed, -self.max_motor_velocity, self.max_motor_velocity)
        right_speed = np.clip(right_speed, -self.max_motor_velocity, self.max_motor_velocity)
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

        return True
        
    def generate_boxes(self):
        N = 7
    
        for i in range(N):
            position = random_position(0.8, 1.2, 0.1)  # Fixed radius range
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
        for i in range(children_field.getCount() - 1, -1, -1):  # Iterate backwards
            node = children_field.getMFNode(i)
            if node and node.getTypeName() == "Solid" and node.getDef() and node.getDef().startswith("WHITE_BOX_"):
                children_field.removeMF(i)

    def reset(self):
        self.robot_node.resetPhysics()
        initial_pos = [np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2), 0.02]
        initial_rot = random_orientation()
        self.translation_field.setSFVec3f(initial_pos)
        self.rotation_field.setSFRotation(initial_rot)

        self.supervisor.step(self.basic_timestep)

        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        
        # Generate boxes once at the beginning
        self.generate_boxes()
        print("Generated obstacle boxes")

    def _load_weights_from_file(self, filepath):
        loaded_weights = np.load(filepath)
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