import numpy as np
from controller import Supervisor
import os
import time
from collections import deque
import math

TIME_STEP = 5
EVALUATION_TIME = 100
RANGE = 7

INPUT = 2
HIDDEN = 4
OUTPUT = 2

BEST_WEIGHTS_FILE = "best_weights_ann_simple.npy"

def random_orientation():
    angle = np.random.uniform(0, 2 * np.pi)
    return [0, 0, 1, angle]

class SimpleANN:
    def __init__(self, weights):
        idx_w1 = HIDDEN * (INPUT + 1)
        self.w1 = weights[:idx_w1].reshape(HIDDEN, INPUT + 1)
        self.w2 = weights[idx_w1:].reshape(OUTPUT, HIDDEN + 1)

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, inputs):
        inputs_with_bias = np.append(inputs, 1.0)
        hidden_inputs = np.dot(self.w1, inputs_with_bias)
        hidden_outputs = self.tanh(hidden_inputs)
        hidden_with_bias = np.append(hidden_outputs, 1.0)
        final_inputs = np.dot(self.w2, hidden_with_bias)
        final_outputs = self.tanh(final_inputs)
        motor_outputs = final_outputs * RANGE

        return motor_outputs

class RunBestSolution:
    def __init__(self):
        self.supervisor = Supervisor()
        self.robot_node = self.supervisor.getFromDef("ROBOT")
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
            if sensor:
                sensor.enable(self.timestep)
                self.ground_sensors.append(sensor)
            else:
                print("sensor missing")


        self.prev_position = self.translation_field.getSFVec3f()

        self.initial_orientation = None

    def runStepLogic(self, ann_model):
        raw_ground_sensor_values = [sensor.getValue() for sensor in self.ground_sensors]

        gs_max_value = 1023.0
        normalized_inputs = [(value / gs_max_value) * 2.0 - 1.0 for value in raw_ground_sensor_values]
        inputs = np.array(normalized_inputs)

        motor_speeds = ann_model.forward(inputs)
        left_speed, right_speed = float(motor_speeds[0]), float(motor_speeds[1])

        left_speed = np.clip(left_speed, -self.max_motor_velocity, self.max_motor_velocity)
        right_speed = np.clip(right_speed, -self.max_motor_velocity, self.max_motor_velocity)
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

        return True

    def reset(self):
        self.robot_node.resetPhysics()
        initial_pos = [0, 0, 0.02]
        initial_rot = random_orientation()
        self.translation_field.setSFVec3f(initial_pos)
        self.rotation_field.setSFRotation(initial_rot)
        self.supervisor.step(self.basic_timestep)

        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
     
        self.initial_orientation = self.rotation_field.getSFRotation()

    def _load_weights_from_file(self, filepath):
        if not os.path.exists(filepath):
            return None

        try:
            loaded_weights = np.load(filepath)
            expected_size = HIDDEN * (INPUT + 1) + OUTPUT * (HIDDEN + 1)
            if len(loaded_weights) == expected_size:
                print("weights loaded")
                return loaded_weights
            else:
                print(f"weight size mismatch: expected {expected_size}, got {len(loaded_weights)}")
                return None
        except Exception as e:
            print(f"weight load error: {e}")
            return None

    def run_best_solution(self, weights_file=None):
        weights_to_run = None

        if weights_file and os.path.exists(weights_file):
             print(f"load weights {weights_file}")
             weights_to_run = self._load_weights_from_file(weights_file)
        elif os.path.exists(BEST_WEIGHTS_FILE):
             print(f"load default {BEST_WEIGHTS_FILE}")
             weights_to_run = self._load_weights_from_file(BEST_WEIGHTS_FILE)

        if weights_to_run is None:
            print("no weights found for visualization")
            return

        print("start visualization")
        self.supervisor.simulationSetMode(self.supervisor.SIMULATION_MODE_REAL_TIME)
        self.supervisor.step(self.basic_timestep)

        self.reset()
        self.prev_position = self.translation_field.getSFVec3f()
        viz_duration = EVALUATION_TIME

        best_ann = SimpleANN(weights_to_run)

        print(f"visualization duration {viz_duration}s")
        start_viz_time = self.supervisor.getTime()

        while True:
            current_time = self.supervisor.getTime()
            if current_time - start_viz_time >= viz_duration:
                 print("visualization time complete")
                 break
            if self.supervisor.step(self.timestep) == -1:
                 print("supervisor closed")
                 break
            self.runStepLogic(best_ann)




if __name__ == "__main__":
    run_controller = RunBestSolution()
    run_controller.run_best_solution()