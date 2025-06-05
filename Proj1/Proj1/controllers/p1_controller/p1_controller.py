import numpy as np
from controller import Supervisor
import random
import math
import csv
import os
from collections import deque

TIME_STEP = 5
POPULATION_SIZE = 10
PARENTS_KEEP = 3
GENERATIONS = 300
MUTATION_RATE = 0.4
MUTATION_SIZE = 8
EVALUATION_TIME = 100
GENOME_SIZE = 6
STAGNATION = 10

W_AREA_COVERAGE = 5.0
W_TIME_ON_LINE = 0.008
W_BACKWARD_PENALTY = -0.006
W_STEPS_SURVIVED_REWARD = 0.001
W_COLLISION_PENALTY = 0

GROUND_SENSOR_THRESHOLD = 200

STANDSTILL_CHECK_STEPS = 10
STANDSTILL_POS_THRESHOLD = 0.005
MAX_STANDSTILL_COUNT_FOR_EARLY_EXIT = 300

CSV_FILENAME = "evolution_log_direct_weights.csv"
WEIGHTS_DIR = "generation_weights_direct"
BEST_WEIGHTS_FILE = "best_weights_direct.npy"


def random_orientation():
    """Gera uma orientação aleatória para o robô."""
    angle = np.random.uniform(0, 2*np.pi)
    return [0, 0, 1, angle]

class Evolution:
    def __init__(self):
        """Inicializa o sistema de evolução com supervisor e configurações do robô."""
        self.supervisor = Supervisor()
        self.robot = self.supervisor.getSelf()
        self.robot_node = self.supervisor.getFromDef("ROBOT") 
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")

        self.basic_timestep = int(self.supervisor.getBasicTimeStep())
        self.timestep = self.basic_timestep * TIME_STEP
        self.supervisor.simulationSetMode(self.supervisor.SIMULATION_MODE_FAST)
        
        self.left_motor = self.supervisor.getDevice('motor.left')
        self.right_motor = self.supervisor.getDevice('motor.right')
        self.ground_sensors = [self.supervisor.getDevice(f'prox.ground.{i}') for i in range(2)]

        for motor in [self.left_motor, self.right_motor]:
            motor.setPosition(float('inf'))

        for sensor in self.ground_sensors:
            sensor.enable(self.timestep)

        self.population = [np.random.uniform(0, 10, GENOME_SIZE) for _ in range(POPULATION_SIZE)]
        self.overall_best_fitness = -float('inf')
        self.best_weights = None
        self.stagnation_counter = 0

        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        self._init_csv()

    def reset(self):
        """Reinicia o robô numa posição e orientação aleatórias."""
        self.robot_node.resetPhysics()
        
        initial_pos = [0, 0, 0.02]
        random_rotation = random_orientation()
        
        self.translation_field.setSFVec3f(initial_pos)
        self.rotation_field.setSFRotation(random_rotation)
        self.supervisor.step(self.basic_timestep)
        
        for motor in [self.left_motor, self.right_motor]:
            motor.setVelocity(0)
        
        self.time_on_line = self.steps_survived = self.backward_movement_count = self.standstill_count = self.step_count = 0
        self.prev_position = self.translation_field.getSFVec3f()
        self.position_history = deque(maxlen=STANDSTILL_CHECK_STEPS)
        self.covered_positions = set()
        self.collision = False

    def runStep(self, weights):
        """Executa um passo da simulação com os pesos fornecidos."""
        self.step_count += 1
        
        if self.collision:
            return False
        
        ground_left = (self.ground_sensors[0].getValue() / 1023 - 0.6) / 0.2 > 0.3
        ground_right = (self.ground_sensors[1].getValue() / 1023 - 0.6) / 0.2 > 0.3
        
        left_speed = ground_left * weights[0] + ground_right * weights[1] + weights[2]
        right_speed = ground_left * weights[3] + ground_right * weights[4] + weights[5]
        
        for motor, speed in zip([self.left_motor, self.right_motor], [left_speed, right_speed]):
            motor.setVelocity(max(min(speed, 9), -9))

        current_position = self.translation_field.getSFVec3f()
        dx, dy = current_position[0] - self.prev_position[0], current_position[1] - self.prev_position[1]
        movement_magnitude = math.sqrt(dx*dx + dy*dy)
        
        if movement_magnitude > 1e-3:
            orientation_angle = self.rotation_field.getSFRotation()[3]
            forward_x, forward_y = math.sin(orientation_angle), -math.cos(orientation_angle)
            movement_x, movement_y = dx / movement_magnitude, dy / movement_magnitude
            if (movement_x * forward_x + movement_y * forward_y) < -0.1:
                self.backward_movement_count += 1

        self.position_history.append((current_position[0], current_position[1]))
        if len(self.position_history) == STANDSTILL_CHECK_STEPS:
            old_x, old_y = self.position_history[0]
            curr_x, curr_y = self.position_history[-1]
            pos_change = math.sqrt((curr_x - old_x)**2 + (curr_y - old_y)**2)
            self.standstill_count = self.standstill_count + 1 if pos_change < STANDSTILL_POS_THRESHOLD else 0
        
        if self.standstill_count >= MAX_STANDSTILL_COUNT_FOR_EARLY_EXIT:
            return False

        raw_left, raw_right = self.ground_sensors[0].getValue(), self.ground_sensors[1].getValue()
        is_on_line = (raw_left < GROUND_SENSOR_THRESHOLD or raw_right < GROUND_SENSOR_THRESHOLD)

        if is_on_line and movement_magnitude > 9.4e-3:
            grid_pos = (int(current_position[0] / 0.05), int(current_position[1] / 0.05))
            self.covered_positions.add(grid_pos)
            self.time_on_line += 1

        self.prev_position = current_position
        self.steps_survived += 1
        
        return True

    def evaluate_individual(self, weights):
        """Avalia a aptidão de um indivíduo com base nos pesos fornecidos."""
        self.reset()
        
        max_steps = int(EVALUATION_TIME * 1000 / self.timestep)
        current_steps = 0
        
        while (current_steps < max_steps and not self.collision):
            if self.supervisor.step(self.timestep) == -1 or not self.runStep(weights):
                break
            current_steps += 1

        final_fitness = (len(self.covered_positions) * W_AREA_COVERAGE + 
                        self.time_on_line * W_TIME_ON_LINE + 
                        W_BACKWARD_PENALTY * self.backward_movement_count + 
                        self.steps_survived * W_STEPS_SURVIVED_REWARD)
        
        return {
            'fitness': final_fitness,
            'area_coverage': len(self.covered_positions),
            'time_on_line': self.time_on_line,
            'steps_survived': self.steps_survived,
            'backward_movement_count': self.backward_movement_count,
            'collision': self.collision,
            'standstill_count': 0,
            'collision_time': 0,
            'weights': weights
        }

    def mutate(self, weights):
        """Aplica mutação aos pesos de um indivíduo."""
        new_weights = weights.copy()
        mask = np.random.random(len(new_weights)) < MUTATION_RATE
        new_weights[mask] += np.random.uniform(-MUTATION_SIZE, MUTATION_SIZE, mask.sum())
        return new_weights

    def crossover(self, parent1, parent2):
        """Realiza cruzamento entre dois progenitores usando ponto único."""
        point = random.randint(1, GENOME_SIZE - 1)
        return (np.concatenate((parent1[:point], parent2[point:])),
                np.concatenate((parent2[:point], parent1[point:])))

    def _init_csv(self):
        """Inicializa o ficheiro CSV para registo da evolução."""
        if not os.path.exists(CSV_FILENAME) or os.path.getsize(CSV_FILENAME) == 0:
            with open(CSV_FILENAME, 'w', newline='') as f:
                csv.writer(f).writerow([
                    "Generation", "Best_Fitness", "Avg_Fitness", 
                    "Area_Coverage", "Time_On_Line", "Steps_Survived",
                    "Backward_Movement_Count", "Standstill_Count", 
                    "Collision_Time", "Collision", "Stagnation_Counter"
                ])

    def _log_generation(self, gen, best_result, avg_fitness):
        """Regista os dados de uma geração no ficheiro CSV."""
        with open(CSV_FILENAME, 'a', newline='') as f:
            csv.writer(f).writerow([
                gen + 1, round(best_result['fitness'], 4), round(avg_fitness, 4),
                best_result['area_coverage'], best_result['time_on_line'],
                best_result['steps_survived'], best_result['backward_movement_count'],
                best_result.get('standstill_count', 0), best_result.get('collision_time', 0),
                best_result['collision'], self.stagnation_counter
            ])

    def run_evolution(self):
        """Executa o algoritmo evolutivo principal."""
        print(f"=== Evolution Started: {POPULATION_SIZE} pop, {GENERATIONS} gens ===")

        for generation in range(GENERATIONS):
            print(f"\n>>> Generation {generation + 1}/{GENERATIONS}")
            
            results = []
            for i, weights in enumerate(self.population):
                results.append(self.evaluate_individual(weights))
                if (i + 1) % max(1, POPULATION_SIZE // 3) == 0:
                    print(f" - Evaluated {i + 1}/{POPULATION_SIZE}")
            
            results.sort(key=lambda x: x['fitness'], reverse=True)
            best_result = results[0]
            avg_fitness = np.mean([r['fitness'] for r in results])

            next_population = [r['weights'] for r in results[:PARENTS_KEEP]]
            
            while len(next_population) < POPULATION_SIZE:
                p1, p2 = random.choices([r['weights'] for r in results[:PARENTS_KEEP]], k=2)
                child1, child2 = self.crossover(p1, p2)
                
                for child in [child1, child2]:
                    if len(next_population) < POPULATION_SIZE:
                        next_population.append(self.mutate(child))
            
            self.population = next_population[:POPULATION_SIZE]

            np.save(os.path.join(WEIGHTS_DIR, f"gen_{generation+1:04d}.npy"), best_result['weights'])

            if best_result['fitness'] > self.overall_best_fitness + 1e-3:
                print(f"New overall best fitness: {best_result['fitness']:.2f} (improvement over {self.overall_best_fitness:.2f})")
                self.overall_best_fitness = best_result['fitness']
                self.best_weights = best_result['weights'].copy()
                np.save(BEST_WEIGHTS_FILE, self.best_weights)
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

            self._log_generation(generation, best_result, avg_fitness)
            print(f"Best={best_result['fitness']:.2f}, Avg={avg_fitness:.2f}, "
                  f"Area={best_result['area_coverage']}, Stagnation={self.stagnation_counter}")

            if self.stagnation_counter >= STAGNATION:
                print("Evolution stopped: Stagnation limit reached.")
                break
        
        print(f"\n=== Finished: Best fitness {self.overall_best_fitness:.2f} ===")
        return self.best_weights

if __name__ == "__main__":
    controller = Evolution()
    best_weights = controller.run_evolution()
