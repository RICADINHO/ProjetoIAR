import numpy as np
from controller import Supervisor
import random
import math
import csv
import os
from collections import deque

# Configuration
TIME_STEP = 5
POPULATION_SIZE = 10
PARENTS_KEEP = 3
GENERATIONS = 1000
STAGNATION_LIMIT = 10
MUTATION_RATE = 0.3
MUTATION_SIZE = 2.0
EVALUATION_TIME = 100
RANGE = 9

INPUT, HIDDEN, OUTPUT = 2, 4, 2
GENOME_SIZE = HIDDEN * (INPUT + 1) + OUTPUT * (HIDDEN + 1)
GROUND_SENSOR_THRESHOLD = 200

# Fitness weights
W_AREA_COVERAGE = 5.0
W_TIME_ON_LINE = 0.008
W_BACKWARD_PENALTY = -0.02
W_STEPS_SURVIVED_REWARD = 0.001

# Standstill detection
STANDSTILL_CHECK_STEPS = 2
STANDSTILL_POS_THRESHOLD = 0.005
MAX_STANDSTILL_COUNT = 300

# File paths
CSV_FILENAME = "evolution_log_ann_simple.csv"
WEIGHTS_DIR = "generation_weights_ann_simple"
BEST_WEIGHTS_FILE = "best_weights_ann_simple.npy"

def random_orientation():
    return [0, 0, 1, np.random.uniform(0, 2 * np.pi)]

class SimpleANN:
    def __init__(self, weights):
        idx = HIDDEN * (INPUT + 1)
        self.w1 = weights[:idx].reshape(HIDDEN, INPUT + 1)
        self.w2 = weights[idx:].reshape(OUTPUT, HIDDEN + 1)

    def forward(self, inputs):
        x = np.append(inputs, 1.0)
        x = np.tanh(self.w1 @ x)
        x = np.append(x, 1.0)
        return np.tanh(self.w2 @ x) * RANGE

class Evolution:
    def __init__(self):
        self.supervisor = Supervisor()
        self.robot_node = self.supervisor.getFromDef("ROBOT")
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")
        self.timestep = int(self.supervisor.getBasicTimeStep()) * TIME_STEP
        self.supervisor.simulationSetMode(self.supervisor.SIMULATION_MODE_FAST)

        # Motors
        self.left_motor = self.supervisor.getDevice('motor.left')
        self.right_motor = self.supervisor.getDevice('motor.right')
        for motor in [self.left_motor, self.right_motor]:
            motor.setPosition(float('inf'))
            motor.setVelocity(0)
        self.max_motor_velocity = self.left_motor.getMaxVelocity()

        # Ground sensors
        self.ground_sensors = [self.supervisor.getDevice(f'prox.ground.{i}') for i in range(2)]
        for sensor in self.ground_sensors:
            sensor.enable(self.timestep)

        # Evolution state
        self.population = [np.random.uniform(-2, 2, GENOME_SIZE) for _ in range(POPULATION_SIZE)]
        self.overall_best_fitness = -float('inf')
        self.best_weights = None
        self.stagnation_counter = 0
        self.generation = 0

        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        self._init_csv()

    def runStepLogic(self, ann_model):
        # Get normalized ground sensor inputs
        ground_values = [sensor.getValue() for sensor in self.ground_sensors]
        inputs = np.array([(v / 1023.0) * 2.0 - 1.0 for v in ground_values])

        # Set motor velocities
        motor_speeds = ann_model.forward(inputs)
        for motor, speed in zip([self.left_motor, self.right_motor], motor_speeds):
            motor.setVelocity(np.clip(speed, -self.max_motor_velocity, self.max_motor_velocity))

        # Movement analysis
        current_pos = self.translation_field.getSFVec3f()
        movement = np.array(current_pos[:2]) - np.array(self.prev_position[:2])
        movement_magnitude = np.linalg.norm(movement)

        # Check backward movement
        if movement_magnitude > 1e-3:
            orientation = self.rotation_field.getSFRotation()[3]
            forward_dir = np.array([math.sin(orientation), -math.cos(orientation)])
            if np.dot(movement / movement_magnitude, forward_dir) < -0.1:
                self.backward_movement_count += 1

        # Standstill detection
        self.position_history.append(current_pos[:2])
        if len(self.position_history) == STANDSTILL_CHECK_STEPS:
            pos_change = np.linalg.norm(np.array(self.position_history[-1]) - np.array(self.position_history[0]))
            self.standstill_count = self.standstill_count + 1 if pos_change < STANDSTILL_POS_THRESHOLD else 0
        
        if self.standstill_count >= MAX_STANDSTILL_COUNT:
            return False

        # Line following reward
        is_on_line = any(v < GROUND_SENSOR_THRESHOLD for v in ground_values)
        if is_on_line and movement_magnitude > 9.4e-3:
            grid_pos = tuple(int(current_pos[i] / 0.05) for i in range(2))
            self.covered_positions.add(grid_pos)
            self.time_on_line += 1

        self.prev_position = current_pos
        self.steps_survived += 1
        return True

    def evaluate_individual(self, weights):
        self.reset()
        max_steps = int(EVALUATION_TIME * 1000 / self.timestep)
        ann = SimpleANN(weights)

        for _ in range(max_steps):
            if self.supervisor.step(self.timestep) == -1 or not self.runStepLogic(ann):
                break

        fitness = (len(self.covered_positions) * W_AREA_COVERAGE + 
                  self.time_on_line * W_TIME_ON_LINE + 
                  self.backward_movement_count * W_BACKWARD_PENALTY + 
                  self.steps_survived * W_STEPS_SURVIVED_REWARD)
                  
        return fitness, len(self.covered_positions), self.time_on_line, self.steps_survived, self.backward_movement_count, self.standstill_count

    def _init_csv(self):
        if not os.path.exists(CSV_FILENAME) or os.path.getsize(CSV_FILENAME) == 0:
            with open(CSV_FILENAME, 'w', newline='') as f:
                csv.writer(f).writerow([
                    "Generation", "Best_Fitness", "Avg_Fitness", "Area_Coverage", 
                    "Time_On_Line", "Steps_Survived", "Backward_Movement_Count", 
                    "Standstill_Count", "Stagnation_Counter"
                ])

    def _log_generation_to_csv(self, data):
        with open(CSV_FILENAME, 'a', newline='') as f:
            csv.writer(f).writerow(data)

    def mutate(self, weights):
        new_weights = weights.copy()
        mask = np.random.random(len(new_weights)) < MUTATION_RATE
        new_weights[mask] += np.random.uniform(-MUTATION_SIZE, MUTATION_SIZE, mask.sum())
        return new_weights

    def crossover(self, parent1, parent2):
        point = random.randint(1, GENOME_SIZE - 1)
        return (np.concatenate((parent1[:point], parent2[point:])),
                np.concatenate((parent2[:point], parent1[point:])))

    def reset(self):
        self.robot_node.resetPhysics()
        initial_pos = [0, 0, 0.02]
        self.translation_field.setSFVec3f(initial_pos)
        self.rotation_field.setSFRotation(random_orientation())
        self.supervisor.step(int(self.supervisor.getBasicTimeStep()))

        for motor in [self.left_motor, self.right_motor]:
            motor.setVelocity(0)

        # Reset metrics
        self.time_on_line = self.steps_survived = self.backward_movement_count = self.standstill_count = 0
        self.position_history = deque(maxlen=STANDSTILL_CHECK_STEPS)
        self.covered_positions = set()
        self.prev_position = self.translation_field.getSFVec3f()

    def run_evolution(self):
        print(f"=== Evolution Started: {POPULATION_SIZE} pop, {GENERATIONS} gen, ANN({INPUT},{HIDDEN},{OUTPUT}) ===")

        for gen in range(GENERATIONS):
            self.generation = gen
            print(f"\n>>> Generation {gen + 1}/{GENERATIONS}")

            # Evaluate population
            results = []
            for i, weights in enumerate(self.population):
                fitness, *metrics = self.evaluate_individual(weights)
                results.append({'fitness': fitness, 'weights': weights, 'metrics': metrics})
                
                if (i + 1) % max(1, POPULATION_SIZE // 3) == 0:
                    print(f" - Evaluated {i + 1}/{POPULATION_SIZE}")

            results.sort(key=lambda x: x['fitness'], reverse=True)

            # Selection and reproduction
            parents = [r['weights'] for r in results[:PARENTS_KEEP]]
            next_population = parents.copy()
            
            # Generate offspring through crossover and mutation
            while len(next_population) < POPULATION_SIZE:
                p1, p2 = random.choices(parents, k=2)
                offspring1, offspring2 = self.crossover(p1, p2)
                for child in [offspring1, offspring2]:
                    if len(next_population) < POPULATION_SIZE:
                        next_population.append(self.mutate(child))

            self.population = next_population

            # Logging
            best_result = results[0]
            best_fitness = best_result['fitness']
            avg_fitness = np.mean([r['fitness'] for r in results if np.isfinite(r['fitness'])])

            np.save(f"{WEIGHTS_DIR}/gen_{gen + 1:04d}.npy", best_result['weights'])

            if best_fitness > self.overall_best_fitness + 1e-3:
                print(f" >> New Best: {best_fitness:.2f} (was {self.overall_best_fitness:.2f})")
                self.overall_best_fitness = best_fitness
                self.best_weights = best_result['weights'].copy()
                np.save(BEST_WEIGHTS_FILE, self.best_weights)
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            # Log to CSV
            log_data = [gen + 1, round(best_fitness, 4), round(avg_fitness, 4)] + best_result['metrics'] + [self.stagnation_counter]
            self._log_generation_to_csv(log_data)

            print(f"   Best: {best_fitness:.2f} | Avg: {avg_fitness:.2f} | Stag: {self.stagnation_counter}")
            print(f"   Stats: Area={log_data[3]}, Line={log_data[4]}, Steps={log_data[5]}, Back={log_data[6]}")

            if self.stagnation_counter >= STAGNATION_LIMIT:
                print("\n--- Stagnation limit reached ---")
                break

        print(f"\n=== Evolution Complete: Best fitness {self.overall_best_fitness:.2f} ===")
        return self.best_weights

if __name__ == "__main__":
    evolution_controller = Evolution()
    best_genome = evolution_controller.run_evolution()
    print("Training complete.")