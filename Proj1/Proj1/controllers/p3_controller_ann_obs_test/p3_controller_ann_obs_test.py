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
HIDDEN = 7
OUTPUT = 2

GROUND_SENSOR_THRESHOLD = 200
PROXIMITY_SENSOR_THRESHOLD_FOR_COLLISION = 2000

STANDSTILL_CHECK_STEPS = 2
STANDSTILL_POS_THRESHOLD = 0.005

BEST_WEIGHTS_FILE = "best_weights_obstacules.npy"
WEIGHTS_DIR_FOR_VIS = "generation_weights_obstacules"

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

class VisualizeSolution:
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

        self.time_on_line = 0
        self.steps_survived = 0
        self.prev_position = None
        self.total_net_displacement_vector = np.array([0.0, 0.0])
        self.backward_movement_count = 0
        self.standstill_count = 0
        self.position_history = deque(maxlen=STANDSTILL_CHECK_STEPS)
        self.initial_orientation = None
        self.collisions_count = 0

    def runStepLogic(self, ann_model):
        step_has_collision = False
        for sensor in self.collision_proximity_sensors:
            if sensor.getValue() > PROXIMITY_SENSOR_THRESHOLD_FOR_COLLISION:
                step_has_collision = True
                break
        if step_has_collision:
            self.collisions_count += 1

        raw_ground_sensor_values = [gs.getValue() for gs in self.ground_sensors]
        gs_max_value = 1023.0
        normalized_ground_inputs = [(val / gs_max_value) * 2.0 - 1.0 for val in raw_ground_sensor_values]

        raw_ann_proximity_values = [ps.getValue() for ps in self.ann_proximity_sensors]
        prox_max_value = 4230.0
        normalized_ann_proximity_inputs = [(val / prox_max_value) * 2.0 - 0.5 for val in raw_ann_proximity_values]

        ann_inputs = np.concatenate((np.array(normalized_ground_inputs), np.array(normalized_ann_proximity_inputs)))

        motor_speeds = ann_model.forward(ann_inputs)
        left_speed, right_speed = float(motor_speeds[0]), float(motor_speeds[1])

        left_speed = np.clip(left_speed, -self.max_motor_velocity, self.max_motor_velocity)
        right_speed = np.clip(right_speed, -self.max_motor_velocity, self.max_motor_velocity)
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

        current_position_vec3 = self.translation_field.getSFVec3f()
        current_position = np.array([current_position_vec3[0], current_position_vec3[1]])

        if self.prev_position is None:
             self.prev_position = current_position

        delta_position = current_position - self.prev_position

        current_rotation_vec4 = self.rotation_field.getSFRotation()
        orientation_angle = current_rotation_vec4[3]

        robot_forward_direction = np.array([math.sin(orientation_angle), -math.cos(orientation_angle)])

        movement_magnitude = np.linalg.norm(delta_position)
        if movement_magnitude > 1e-4:
            movement_direction = delta_position / movement_magnitude
            dot_product = np.dot(movement_direction, robot_forward_direction)
            if dot_product < -0.1:
                self.backward_movement_count += 1

        self.position_history.append(current_position.copy())
        if len(self.position_history) == STANDSTILL_CHECK_STEPS:
            pos_change = np.linalg.norm(self.position_history[-1] - self.position_history[0])
            if pos_change < STANDSTILL_POS_THRESHOLD:
                self.standstill_count += 1

        is_on_line = (raw_ground_sensor_values[0] < GROUND_SENSOR_THRESHOLD or
                      raw_ground_sensor_values[1] < GROUND_SENSOR_THRESHOLD)
        if is_on_line:
            self.time_on_line += 1
            self.total_net_displacement_vector += delta_position

        self.prev_position = current_position
        self.steps_survived += 1
        return True

    def reset(self):
        self.robot_node.resetPhysics()
        initial_pos = [np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2), 0.02]
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
        self.collisions_count = 0
        self.position_history.clear()

        current_pos_vec3 = self.translation_field.getSFVec3f()
        self.prev_position = np.array([current_pos_vec3[0], current_pos_vec3[1]])
        self.initial_orientation = self.rotation_field.getSFRotation()


    def _load_weights_from_file(self, filepath):
        loaded_weights = np.load(filepath)
        return loaded_weights

    def run_visualization(self, weights_file=None):
        weights_to_run = None

        if weights_file:
            weights_to_run = self._load_weights_from_file(weights_file)
        else:
            weights_to_run = self._load_weights_from_file(BEST_WEIGHTS_FILE)

        self.supervisor.simulationSetMode(self.supervisor.SIMULATION_MODE_REAL_TIME)
        self.supervisor.step(self.basic_timestep)

        self.reset()

        ann_instance = SimpleANN(weights_to_run)

        max_sim_steps = int((EVALUATION_TIME * 1000) / self.timestep)

        start_time = self.supervisor.getTime()

        while self.supervisor.step(self.timestep) != -1:
            current_sim_time = self.supervisor.getTime()
            if current_sim_time - start_time >= EVALUATION_TIME:
                break

            self.runStepLogic(ann_instance)


        net_displacement_magnitude = np.linalg.norm(self.total_net_displacement_vector)
        print(f"Steps: {self.steps_survived}")
        print(f"Time on line: {self.time_on_line} steps")
        print(f"Net displacement: {net_displacement_magnitude:.3f}")
        print(f"Standstill count: {self.standstill_count}")
        print(f"Backward count: {self.backward_movement_count}")
        print(f"Collision steps: {self.collisions_count}")


if __name__ == "__main__":
    visualizer = VisualizeSolution()

    visualizer.run_visualization()