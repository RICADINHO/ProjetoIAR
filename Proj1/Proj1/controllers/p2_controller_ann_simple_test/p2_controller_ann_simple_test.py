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

GROUND_SENSOR_THRESHOLD = 200

STANDSTILL_CHECK_STEPS = 2
STANDSTILL_POS_THRESHOLD = 0.005

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

        self.time_on_line = 0
        self.steps_survived = 0
        self.prev_position = self.translation_field.getSFVec3f()
        self.total_net_displacement_vector = np.array([0.0, 0.0])
        self.backward_movement_count = 0
        self.standstill_count = 0
        self.position_history = deque(maxlen=STANDSTILL_CHECK_STEPS)
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

        current_position = self.translation_field.getSFVec3f()
        dx = current_position[0] - self.prev_position[0]
        dy = current_position[1] - self.prev_position[1]
        current_orientation = self.rotation_field.getSFRotation()
        orientation_angle = current_orientation[3]

        forward_x = math.sin(orientation_angle)
        forward_y = -math.cos(orientation_angle)
        forward_direction = np.array([forward_x, forward_y])
        movement_direction = np.array([dx, dy])
        movement_magnitude = np.linalg.norm(movement_direction)
        if movement_magnitude > 1e-3:
            normalized_movement = movement_direction / movement_magnitude
            normalized_forward = forward_direction / np.linalg.norm(forward_direction)
            dot_product = np.dot(normalized_movement, normalized_forward)
            if dot_product < 0:
                self.backward_movement_count += 1

        self.position_history.append((current_position[0], current_position[1], orientation_angle))
        if len(self.position_history) == STANDSTILL_CHECK_STEPS:
            oldest_pos_x, oldest_pos_y, _ = self.position_history[0]
            current_pos_x, current_pos_y, _ = self.position_history[-1]

            pos_change_over_time = np.linalg.norm(np.array([current_pos_x, current_pos_y]) - np.array([oldest_pos_x, oldest_pos_y]))

            if pos_change_over_time < STANDSTILL_POS_THRESHOLD:
                self.standstill_count += 1

        is_on_line = (raw_ground_sensor_values[0] < GROUND_SENSOR_THRESHOLD or
                      raw_ground_sensor_values[1] < GROUND_SENSOR_THRESHOLD)

        if is_on_line:
            self.time_on_line += 1
            self.total_net_displacement_vector += np.array([dx, dy])

        self.prev_position = current_position
        self.steps_survived += 1
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
        self.time_on_line = 0
        self.steps_survived = 0
        self.total_net_displacement_vector = np.array([0.0, 0.0])
        self.backward_movement_count = 0
        self.standstill_count = 0
        self.position_history.clear()
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

        print("visualization stats")
        net_displacement_magnitude = np.linalg.norm(self.total_net_displacement_vector)
        print(f"steps completed: {self.steps_survived}")
        print(f"time on line: {self.time_on_line}")
        print(f"net displacement: {net_displacement_magnitude:.1f}")
        print(f"standstill count: {self.standstill_count}")
        print("---")


if __name__ == "__main__":
    run_controller = RunBestSolution()
    run_controller.run_best_solution()