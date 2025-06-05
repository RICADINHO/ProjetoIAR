import numpy as np
from controller import Supervisor
import random
import math
import csv
import os
import time
from collections import deque

# Configuration
TIME_STEP = 5
POPULATION_SIZE = 10
PARENTS_KEEP = 3
GENERATIONS = 1000
STAGNATION_LIMIT = 30
MUTATION_RATE = 0.3
MUTATION_SIZE = 3
EVALUATION_TIME = 100

# Neural network
INPUT, HIDDEN1, HIDDEN2, OUTPUT = 5, 11, 7, 2
GENOME_SIZE = HIDDEN1*(INPUT+1) + HIDDEN2*(HIDDEN1+1) + OUTPUT*(HIDDEN2+1)

# Fitness weights
GROUND_THRESHOLD = 200
W_DISPLACEMENT, W_TIME_ON_LINE, W_BACKWARD, W_COLLISION = 3.0, 0.008, -0.006, -0.001
STANDSTILL_THRESHOLD = 0.003

# Files
CSV_FILE = "evolution_log_obstacles.csv"
WEIGHTS_DIR = "generation_weights_obstacles"
BEST_WEIGHTS_FILE = "best_weights_obstacles.npy"

def random_position(min_r, max_r, z):
    r = np.sqrt(np.random.uniform(min_r**2, max_r**2))
    theta = np.random.uniform(0, 2*np.pi)
    return (r*np.cos(theta), r*np.sin(theta), z)

def random_orientation():
    return [0, 0, 1, np.random.uniform(0, 2*np.pi)]

class SimpleANN:
    def __init__(self, weights):
        i1 = HIDDEN1 * (INPUT + 1)
        i2 = i1 + HIDDEN2 * (HIDDEN1 + 1)
        self.w1 = weights[:i1].reshape(HIDDEN1, INPUT + 1)
        self.w2 = weights[i1:i2].reshape(HIDDEN2, HIDDEN1 + 1)
        self.w3 = weights[i2:].reshape(OUTPUT, HIDDEN2 + 1)
    
    def forward(self, inputs):
        h1 = np.tanh(np.dot(self.w1, np.append(inputs, 1.0)))
        h2 = np.tanh(np.dot(self.w2, np.append(h1, 1.0)))
        return np.tanh(np.dot(self.w3, np.append(h2, 1.0))) * 9

class Evolution:
    def __init__(self):
        self.supervisor = Supervisor()
        self.root = self.supervisor.getRoot()
        self.robot_node = self.supervisor.getFromDef("ROBOT")
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")
        
        self.timestep = int(self.supervisor.getBasicTimeStep()) * TIME_STEP
        self.basic_timestep = int(self.supervisor.getBasicTimeStep())
        self.supervisor.simulationSetMode(self.supervisor.SIMULATION_MODE_FAST)

        # Motors
        self.left_motor = self.supervisor.getDevice('motor.left')
        self.right_motor = self.supervisor.getDevice('motor.right')
        for motor in [self.left_motor, self.right_motor]:
            motor.setPosition(float('inf'))
            motor.setVelocity(0)
        self.max_vel = self.left_motor.getMaxVelocity()

        # Sensors
        self.prox_sensors = [self.supervisor.getDevice(f'prox.horizontal.{i}') for i in range(7)]
        self.ground_sensors = [self.supervisor.getDevice(f'prox.ground.{i}') for i in range(2)]
        
        for sensor in self.prox_sensors + self.ground_sensors:
            sensor.enable(self.timestep)

        # Evolution state
        self.population = [np.random.uniform(-2, 2, GENOME_SIZE) for _ in range(POPULATION_SIZE)]
        self.best_fitness = -float('inf')
        self.best_weights = None
        self.stagnation_counter = 0
        self.generation = 0
        
        self.generate_boxes()
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        self._init_csv()

    def generate_boxes(self):
        """Generate random obstacle boxes in the environment"""
        for i in range(6):
            pos = random_position(0.4, 1.45, 1)
            orient = random_orientation()
            length = np.random.uniform(0.05, 0.2)
            width = np.random.uniform(0.05, 0.2)
            
            box_string = f"""
            DEF WHITE_BOX_{i} Solid {{
              translation {pos[0]} {pos[1]} {pos[2]}
              rotation {orient[0]} {orient[1]} {orient[2]} {orient[3]}
              physics Physics {{ density 1000.0 }}
              children [
                Shape {{
                  appearance Appearance {{
                    material Material {{ diffuseColor 1 1 1 }}
                  }}
                  geometry Box {{ size {length} {width} 0.2 }}
                }}
              ]
              boundingObject Box {{ size {length} {width} 0.2 }}
            }}"""
            self.root.getField('children').importMFNodeFromString(-1, box_string)

    def del_boxes(self):
        """Remove all boxes from the scene"""
        for i in range(self.root.getField('children').getCount()):
            node = self.root.getField('children').getMFNode(i)
            if (node.getTypeName() == "Solid" and node.getDef() and 
                node.getDef().startswith("WHITE_BOX_")):
                self.root.getField('children').removeMF(i)
                return self.del_boxes()  # Recursive call due to index changes

    def get_sensor_data(self):
        # Proximity sensors (front, center-front, center)
        prox = [self.prox_sensors[i].getValue() for i in [0, 2, 4]]
        prox_norm = [(v / 4230.0) * 2.0 - 1.0 for v in prox]
        
        # Ground sensors
        ground_norm = [(s.getValue() / 1023.0) * 2.0 - 1.0 for s in self.ground_sensors]
        
        return np.array(prox_norm + ground_norm)

    def evaluate_individual(self, weights):
        self.reset_robot()
        
        # Metrics
        time_on_line = backward_count = collision_time = standstill_count = 0
        covered_positions = set()
        prev_pos = None
        position_history = deque(maxlen=2)
        
        ann = SimpleANN(weights)
        max_steps = int(EVALUATION_TIME * 1000 / self.timestep)
        
        for step in range(max_steps):
            if self.supervisor.step(self.timestep) == -1:
                break
                
            # Control robot
            inputs = self.get_sensor_data()
            velocities = ann.forward(inputs)
            
            left_vel = np.clip(velocities[0], -self.max_vel, self.max_vel)
            right_vel = np.clip(velocities[1], -self.max_vel, self.max_vel)
            self.left_motor.setVelocity(left_vel)
            self.right_motor.setVelocity(right_vel)
            
            current_pos = self.translation_field.getSFVec3f()
            
            # Check backward movement
            if prev_pos:
                dx, dy = current_pos[0] - prev_pos[0], current_pos[1] - prev_pos[1]
                if np.linalg.norm([dx, dy]) > 1e-3:
                    rotation = self.rotation_field.getSFRotation()
                    forward = np.array([math.sin(rotation[3]), -math.cos(rotation[3])])
                    if np.dot([dx, dy], forward) < 0:
                        backward_count += 1
            
            # Track position history for standstill detection
            position_history.append((current_pos[0], current_pos[1]))
            if len(position_history) >= 2:
                pos_change = np.linalg.norm(np.array(position_history[-1]) - np.array(position_history[0]))
                if pos_change < STANDSTILL_THRESHOLD:
                    standstill_count += 1
                    if standstill_count >= 300:  # Too long without movement
                        break
            
            # Time on line and area coverage
            ground_values = [s.getValue() for s in self.ground_sensors]
            if any(v < GROUND_THRESHOLD for v in ground_values):
                if np.linalg.norm([dx if prev_pos else 0, dy if prev_pos else 0]) > 9.4e-3:
                    grid_pos = (int(current_pos[0] / 0.05), int(current_pos[1] / 0.05))
                    covered_positions.add(grid_pos)
                    time_on_line += 1
            
            # Collision detection
            if any(s.getValue() > 2000 for s in self.prox_sensors):
                collision_time += 1
            
            prev_pos = current_pos
        
        # Calculate fitness
        fitness = (len(covered_positions) * W_DISPLACEMENT +
                  time_on_line * W_TIME_ON_LINE +
                  backward_count * W_BACKWARD +
                  collision_time * W_COLLISION * (self.generation + 1) * 0.1 +
                  step * 0.001)
        
        return (fitness, len(covered_positions), time_on_line, step, 
                backward_count, standstill_count, collision_time)

    def reset_robot(self):
        self.robot_node.resetPhysics()
        self.translation_field.setSFVec3f([0, 0, 0.02])
        self.rotation_field.setSFRotation(random_orientation())
        self.supervisor.step(self.basic_timestep)
        for motor in [self.left_motor, self.right_motor]:
            motor.setVelocity(0)

    def mutate(self, weights):
        new_weights = weights.copy()
        mask = np.random.random(len(new_weights)) < MUTATION_RATE
        new_weights[mask] += np.random.uniform(-MUTATION_SIZE, MUTATION_SIZE, np.sum(mask))
        return new_weights

    def _init_csv(self):
        if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
            with open(CSV_FILE, 'w', newline='') as f:
                csv.writer(f).writerow([
                    "Generation", "Best_Fitness", "Avg_Fitness", "Area_Coverage",
                    "Time_On_Line", "Steps_Survived", "Backward_Movement_Count",
                    "Standstill_Count", "Collision_Time", "Stagnation_Counter"
                ])

    def run_evolution(self):
        print(f"Starting evolution: {POPULATION_SIZE} individuals, {GENERATIONS} generations")
        
        for generation in range(GENERATIONS):
            self.generation = generation
            print(f"\nGeneration {generation + 1}/{GENERATIONS}")
            
            # Regenerate obstacles each generation
            self.del_boxes()
            self.generate_boxes()
            
            # Evaluate population
            results = []
            for i, weights in enumerate(self.population):
                fitness, area, time_line, steps, backward, standstill, collisions = self.evaluate_individual(weights)
                results.append({
                    'fitness': fitness, 'area': area, 'time_line': time_line,
                    'steps': steps, 'backward': backward, 'standstill': standstill,
                    'collisions': collisions, 'weights': weights
                })
                
                if (i + 1) % max(1, POPULATION_SIZE // 3) == 0:
                    print(f"  Evaluated {i + 1}/{POPULATION_SIZE}")
            
            # Sort by fitness
            results.sort(key=lambda x: x['fitness'], reverse=True)
            best_result = results[0]
            best_fitness = best_result['fitness']
            avg_fitness = np.mean([r['fitness'] for r in results if np.isfinite(r['fitness'])])
            
            # Save generation best
            np.save(os.path.join(WEIGHTS_DIR, f"gen_{generation + 1:04d}.npy"), best_result['weights'])
            
            # Check for improvement
            if best_fitness > self.best_fitness + 0.1:
                print(f"  New best fitness: {best_fitness:.2f}")
                self.best_fitness = best_fitness
                self.best_weights = best_result['weights'].copy()
                np.save(BEST_WEIGHTS_FILE, self.best_weights)
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            
            # Log results
            log_data = [
                generation + 1, round(best_fitness, 4), round(avg_fitness, 4),
                best_result['area'], best_result['time_line'], best_result['steps'],
                best_result['backward'], best_result['standstill'], best_result['collisions'],
                self.stagnation_counter
            ]
            
            with open(CSV_FILE, 'a', newline='') as f:
                csv.writer(f).writerow(log_data)
            
            print(f"  Best: {best_fitness:.2f} | Avg: {avg_fitness:.2f} | Stagnation: {self.stagnation_counter}")
            print(f"  Area: {best_result['area']} | Steps: {best_result['steps']} | Collisions: {best_result['collisions']}")
            
            # Check stagnation
            if self.stagnation_counter >= STAGNATION_LIMIT:
                print("Stagnation limit reached!")
                break
            
            # Create new population
            self.population = [r['weights'] for r in results[:PARENTS_KEEP]]
            while len(self.population) < POPULATION_SIZE:
                parent = random.choice(self.population[:PARENTS_KEEP])
                self.population.append(self.mutate(parent))
        
        print(f"\nEvolution complete! Best fitness: {self.best_fitness:.2f}")
        return self.best_weights

if __name__ == "__main__":
    evolution = Evolution()
    best_weights = evolution.run_evolution()
    print("Training complete!")