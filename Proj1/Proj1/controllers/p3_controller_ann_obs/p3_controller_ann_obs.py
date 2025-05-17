import numpy as np
from controller import Supervisor
import random
import math
import csv
import os
import time
from collections import deque

TIME_STEP = 5
POPULATION_SIZE = 10
PARENTS_KEEP = 3
GENERATIONS = 1000
STAGNATION_LIMIT = 10
MUTATION_RATE = 0.6
MUTATION_SIZE = 0.3
EVALUATION_TIME = 130
RANGE = 9

INPUT = 5
HIDDEN = 7
OUTPUT = 2
GENOME_SIZE = HIDDEN * (INPUT + 1) + OUTPUT * (HIDDEN + 1)

GROUND_SENSOR_THRESHOLD = 200

W_NET_DISPLACEMENT = 100.0
W_TIME_ON_LINE = 0.1
W_BACKWARD_PENALTY = -0.1
W_STANDSTILL_PENALTY = -0.1
W_COLLISION = -0.01

STANDSTILL_CHECK_STEPS = 2
STANDSTILL_POS_THRESHOLD = 0.005

CSV_FILENAME = "evolution_log_obstacules.csv"
WEIGHTS_DIR = "generation_weights_obstacules"
BEST_WEIGHTS_FILE = "best_weights_obstacules.npy"

def random_orientation():
    angle = np.random.uniform(0, 2 * np.pi)
    return [0, 0, 1, angle]

def angular_distance(angle1, angle2):
    delta_angle = abs(angle1 - angle2)
    delta_angle = delta_angle % (2 * math.pi)
    return min(delta_angle, 2 * math.pi - delta_angle)

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

class Evolution:
    def __init__(self):
        self.collision = False
        self.supervisor = Supervisor()
        self.robot_node = self.supervisor.getFromDef("ROBOT")
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")
        self.basic_timestep = int(self.supervisor.getBasicTimeStep())
        self.timestep = self.basic_timestep * TIME_STEP
        self.supervisor.simulationSetMode(self.supervisor.SIMULATION_MODE_FAST)

        self.left_motor = self.supervisor.getDevice('motor.left')
        self.right_motor = self.supervisor.getDevice('motor.right')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.max_motor_velocity = self.left_motor.getMaxVelocity()

        self.__ir_0 = self.supervisor.getDevice('prox.horizontal.0')
        self.__ir_1 = self.supervisor.getDevice('prox.horizontal.1')
        self.__ir_2 = self.supervisor.getDevice('prox.horizontal.2')
        self.__ir_3 = self.supervisor.getDevice('prox.horizontal.3')
        self.__ir_4 = self.supervisor.getDevice('prox.horizontal.4')
        self.__ir_5 = self.supervisor.getDevice('prox.horizontal.5')
        self.__ir_6 = self.supervisor.getDevice('prox.horizontal.6')
        self.__ir_7 = self.supervisor.getDevice('prox.ground.0')
        self.__ir_8 = self.supervisor.getDevice('prox.ground.1')

        self.__ir_0.enable(self.timestep)
        self.__ir_1.enable(self.timestep)
        self.__ir_2.enable(self.timestep)
        self.__ir_3.enable(self.timestep)
        self.__ir_4.enable(self.timestep)
        self.__ir_5.enable(self.timestep)
        self.__ir_6.enable(self.timestep)
        self.__ir_7.enable(self.timestep)
        self.__ir_8.enable(self.timestep)

        self.sensors = [self.__ir_0, self.__ir_2, self.__ir_4]
        self.ground_sensors = [self.supervisor.getDevice(f'prox.ground.{i}') for i in range(2)]
        self.__n = 0
        
        self.time_on_line = 0
        self.steps_survived = 0
        self.collisions_time = 0
        self.prev_position = self.translation_field.getSFVec3f()
        self.total_net_displacement_vector = np.array([0.0, 0.0])
        self.backward_movement_count = 0
        self.standstill_count = 0
        self.position_history = deque(maxlen=STANDSTILL_CHECK_STEPS)
        self.initial_orientation = None

        self.population = self.create_initial_population()
        self.overall_best_fitness = -float('inf')
        self.best_weights = None
        self.stagnation_counter = 0

        if not os.path.exists(WEIGHTS_DIR):
            os.makedirs(WEIGHTS_DIR)
        self._init_csv()

    def runStepLogic(self, ann_model):
        self.collision = bool(
            (self.__ir_0.getValue() > 3000 or 
             self.__ir_1.getValue() > 3000 or
             self.__ir_2.getValue() > 3000 or
             self.__ir_3.getValue() > 3000 or
             self.__ir_4.getValue() > 3000 or
             self.__ir_5.getValue() > 3000 or
             self.__ir_6.getValue() > 3000)
        )
        if self.collision:
            self.collisions_time += 1
            
        raw_ground_sensor_values = [sensor.getValue() for sensor in self.ground_sensors]
        gs_max_value = 1023.0
        normalized_inputs = [(value / gs_max_value) * 2.0 - 1.0 for value in raw_ground_sensor_values]
        
        proximity_max_value = 4230.0
        proximity_inputs = np.array([self.__ir_0.getValue(), self.__ir_2.getValue(), self.__ir_4.getValue()]).reshape(1, 3)
        normalized_proximity = [(value / proximity_max_value) * 2.0 - 0.5 for value in proximity_inputs]
        
        inputs = np.concatenate((np.array(normalized_inputs), normalized_proximity), axis=None)
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

    def evaluate_individual(self, weights):
        self.reset()
        self.prev_position = self.translation_field.getSFVec3f()
        max_steps = int(EVALUATION_TIME * 1000 / self.timestep)
        individual_ann = SimpleANN(weights)

        while self.steps_survived < max_steps:
            if self.supervisor.step(self.timestep) == -1:
                break
            self.runStepLogic(individual_ann)

        net_displacement_magnitude = np.linalg.norm(self.total_net_displacement_vector)
        backward_penalty = W_BACKWARD_PENALTY * self.backward_movement_count
        standstill_penalty = W_STANDSTILL_PENALTY * self.standstill_count
        collisions_penalty = W_COLLISION * self.collisions_time

        final_fitness = (
            net_displacement_magnitude * W_NET_DISPLACEMENT +
            self.time_on_line * W_TIME_ON_LINE +
            backward_penalty +
            standstill_penalty +
            collisions_penalty
        )

        return final_fitness, net_displacement_magnitude, self.time_on_line, self.standstill_count, self.collisions_time

    def _init_csv(self):
        write_header = not os.path.exists(CSV_FILENAME) or os.path.getsize(CSV_FILENAME) == 0
        with open(CSV_FILENAME, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                header = [
                    "Generation", "Best_Fitness", "Avg_Fitness",
                    "Net_Displacement", "Time_On_Line",
                    "Standstill_Count",
                    "Stagnation_Counter", "Weights_Filename"
                ]
                writer.writerow(header)

    def _log_generation_to_csv(self, generation_data):
        with open(CSV_FILENAME, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(generation_data)

    def create_initial_population(self):
        population = []
        for _ in range(POPULATION_SIZE):
            weights = np.random.uniform(0, 2, GENOME_SIZE)
            population.append(weights)
        return population

    def mutate(self, weights):
        new_weights = weights.copy()
        for i in range(len(new_weights)):
            if random.random() < MUTATION_RATE:
                noise = np.random.normal(0, MUTATION_SIZE)
                new_weights[i] += noise
        return new_weights

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
        self.collisions_time = 0
        self.position_history.clear()
        self.initial_orientation = self.rotation_field.getSFRotation()

    def run_evolution(self):
        print("start evo")
        print(f"pop {POPULATION_SIZE} gens {GENERATIONS} stag {STAGNATION_LIMIT}")
        print("fitness weights set")
        print("ann config ok")
        print("sensor threshold")
        print("standstill config")
        print("---")

        for generation in range(GENERATIONS):
            evaluation_results = []
            print(f"gen {generation + 1}")
            start_gen_time = time.time()

            for i, weights in enumerate(self.population):
                if self.supervisor.simulationGetMode() != self.supervisor.SIMULATION_MODE_FAST:
                    self.supervisor.simulationSetMode(self.supervisor.SIMULATION_MODE_FAST)
                    self.supervisor.step(self.basic_timestep)

                fitness, net_disp, time_on_line, standstill_count, collisions_time = self.evaluate_individual(weights)
                evaluation_results.append({
                    'fitness': fitness,
                    'net_displacement': net_disp,
                    'time_on_line': time_on_line,
                    'standstill_count': standstill_count,
                    'weights': weights
                })

                if (i + 1) % max(1, POPULATION_SIZE // 3) == 0 or i == POPULATION_SIZE - 1:
                    print(f"eval {i+1}/{POPULATION_SIZE}")

            evaluation_results.sort(key=lambda x: x['fitness'], reverse=True)

            # Select top PARENTS_KEEP individuals and generate new population using mutation only
            next_population = [res['weights'] for res in evaluation_results[:PARENTS_KEEP]]
            while len(next_population) < POPULATION_SIZE:
                parent_weights = random.choice(next_population[:PARENTS_KEEP])
                offspring_weights = self.mutate(parent_weights)
                next_population.append(offspring_weights)

            self.population = next_population

            gen_best_result = evaluation_results[0]
            gen_best_fitness = gen_best_result['fitness']
            finite_fitnesses = [res['fitness'] for res in evaluation_results if np.isfinite(res['fitness'])]
            gen_avg_fitness = np.mean(finite_fitnesses) if finite_fitnesses else -float('inf')

            best_gen_weights_filename = os.path.join(WEIGHTS_DIR, f"gen_{generation+1:04d}.npy")
            np.save(best_gen_weights_filename, gen_best_result['weights'])

            improvement_threshold = 1e-3
            if gen_best_fitness > self.overall_best_fitness + improvement_threshold:
                print(f"new best {gen_best_fitness:.2f}")
                self.overall_best_fitness = gen_best_fitness
                self.best_weights = gen_best_result['weights'].copy()
                np.save(BEST_WEIGHTS_FILE, self.best_weights)
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            log_data = [
                generation + 1,
                round(gen_best_fitness, 4) if np.isfinite(gen_best_fitness) else 'NaN',
                round(gen_avg_fitness, 4) if np.isfinite(gen_avg_fitness) else 'NaN',
                round(gen_best_result['net_displacement'], 3),
                gen_best_result['time_on_line'],
                gen_best_result['standstill_count'],
                self.stagnation_counter,
                os.path.basename(best_gen_weights_filename)
            ]
            self._log_generation_to_csv(log_data)

            print(f"gen {generation+1} best {log_data[1]} avg {log_data[2]} stag {self.stagnation_counter}")
            print(f"gen best disp {gen_best_result['net_displacement']:.1f} line {gen_best_result['time_on_line']} still {gen_best_result['standstill_count']}")

            if self.stagnation_counter >= STAGNATION_LIMIT:
                print("stop stag")
                break

        if generation == GENERATIONS - 1:
            print("stop max gen")

        print("evo done")
        if self.best_weights is not None:
            print(f"best total {self.overall_best_fitness:.2f}")
            best_overall_weights_filename = f"best_weights_overall_fitness_{self.overall_best_fitness:.2f}.npy"
            print(f"save weights {best_overall_weights_filename}")
            np.save(BEST_WEIGHTS_FILE, self.best_weights)
        else:
            print("no best found")

        return self.best_weights

if __name__ == "__main__":
    evolution_controller = Evolution()
    best_genome_overall = evolution_controller.run_evolution()
    print("training complete")