import tqdm
import time
import numpy as np 
import matplotlib.pyplot as plt
from .aco_structures import Edge, Ant, Problem

class SolveTSPUsingACO:
    def __init__(self, problem, mode='ACS', colony_size=10,
                 elitist_weight=1.0, min_scaling_factor=0.001,
                 alpha=1.0, beta=3.0, rho=0.1, pheromone_deposit_weight=1.0,
                 initial_pheromone=1.0, steps=100):
        self.problem = problem
        self.mode = mode
        self.colony_size = colony_size
        self.elitist_weight = elitist_weight
        self.min_scaling_factor = min_scaling_factor
        self.rho = rho 
        self.pheromone_deposit_weight = pheromone_deposit_weight
        self.steps = steps
        
        self.num_nodes = problem.num_request + 1
        
        self.edges = [[None]* self.num_nodes for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                weight = problem.time_matrix[i][j]
                self.edges[i][j] = Edge(i, j, weight, initial_pheromone)
        
        self.ants = [Ant(alpha, beta, self.edges, self.problem) for _ in range(self.colony_size)]
        
        self.global_best_tour = None
        self.global_best_cost = float('inf')
        self.global_best_details = {} 
        
    def _add_pheromone(self, tour, cost, weight=1.0):
        additional_pheromone = self.pheromone_deposit_weight / cost
        full_path = [0] + tour + [0]
        for i in range(len(full_path)-1):
            u = full_path[i]
            v = full_path[i+1]
            self.edges[u][v].pheromone += weight * additional_pheromone
    
    def _acs(self):
        progress_best = []
        for step in tqdm.tqdm(range(self.steps), desc=f"Running ACO ({self.mode})"):
            for ant in self.ants:
                ant.find_tour()
                cost = ant.calculate_cost()
                self._add_pheromone(ant.tour, cost)
                if cost < self.global_best_cost:
                    self.global_best_tour = list(ant.tour)
                    self.global_best_cost = cost
                    self.global_best_tour =ant.tour
                    self.global_best_details={"travel": ant.travel_time, "late": ant.total_lateness, "wait": ant.total_wait}
            
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0-self.rho)
            progress_best.append(self.global_best_cost)
        return progress_best
    
    def _elitist(self):
        progress_best = []
        for step in tqdm.tqdm(range(self.steps), desc=f"Running ACO ({self.mode})"):
            for ant in self.ants:
                ant.find_tour()
                cost = ant.calculate_cost()
                self._add_pheromone(ant.tour, cost)
                if cost < self.global_best_cost:
                    self.global_best_tour = list(ant.tour)
                    self.global_best_cost = cost
                    self.global_best_tour =ant.tour
                    self.global_best_details={"travel": ant.travel_time, "late": ant.total_lateness, "wait": ant.total_wait}
            
            if self.global_best_tour:
                self._add_pheromone(self.global_best_tour, self.global_best_cost, weight=self.elitist_weight)
            
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0-self.rho)
            progress_best.append(self.global_best_cost)
        return progress_best

    def _max_min(self):
        progress_best = []
        for step in tqdm.tqdm(range(self.steps), desc=f"Running ACO ({self.mode})"):
            iteration_best_tour = None
            iteration_best_cost = float('inf')
            
            # Tìm đường tốt nhất trong thế hệ này
            for ant in self.ants:
                ant.find_tour()
                cost = ant.calculate_cost()
                if cost < iteration_best_cost:
                    iteration_best_tour = ant.tour
                    iteration_best_cost = cost
                    iteration_best_tour = ant.tour
            
            # Cập nhật Global Best
            if iteration_best_cost < self.global_best_cost:
                self.global_best_tour = list(iteration_best_tour)
                self.global_best_cost = iteration_best_cost
                self.global_best_tour = iteration_best_tour
                # Tìm ant tương ứng để lấy details
                for ant in self.ants:
                     if ant.total_cost == iteration_best_cost:
                        self.global_best_details={"travel": ant.travel_time, "late": ant.total_lateness, "wait": ant.total_wait}
                        break

            # Cập nhật Pheromone
            if float(step + 1) / float(self.steps) <= 0.75:
                self._add_pheromone(iteration_best_tour, iteration_best_cost)
                max_pheromone = self.pheromone_deposit_weight / iteration_best_cost if iteration_best_cost > 0 else 10.0
            else:
                self._add_pheromone(self.global_best_tour, self.global_best_cost)
                max_pheromone = self.pheromone_deposit_weight / self.global_best_cost
            
            min_pheromone = max_pheromone * self.min_scaling_factor
            
            # Bay hơi và kẹp giá trị 
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    self.edges[i][j].pheromone *= (1.0-self.rho)
                    val = self.edges[i][j].pheromone
                    if val > max_pheromone: val = max_pheromone
                    if val < min_pheromone: val = min_pheromone
                    self.edges[i][j].pheromone = val
                    
            progress_best.append(self.global_best_cost)
        return progress_best
    
    def run1(self):
        print(f"Starting ACO algorithm ({self.mode})...")
        start = time.time()
        progress = []
        if self.mode == 'ACS':
            progress = self._acs()
        elif self.mode == 'Elitist':
            progress = self._elitist()
        else:
            progress = self._max_min()
        
        runtime = time.time() - start
        
        print(f"Complete {self.mode}")
        print(f"Runtime: {runtime:.4f}s")
        print(f"Best Fitness (Cost): {self.global_best_cost:.2f}")
        print(f"Travel={self.global_best_details.get('travel',0)}, "
              f"Late={self.global_best_details.get('late',0)}, Wait={self.global_best_details.get('wait',0)}")
        
        plt.plot(progress, label=f'{self.mode}')
        plt.title(f"ACO Convergence (N={self.problem.num_request})")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def run(self):
        print(f"Starting ACO algorithm ({self.mode})...")
        start = time.time()
        progress = []
    
        if self.mode == 'ACS':
            progress = self._acs()
        elif self.mode == 'Elitist':
            progress = self._elitist()
        else:
            progress = self._max_min()
        
        runtime = time.time() - start
        
        print(f"Completed {self.mode}")
        print(f"Runtime: {runtime:.4f}s")
        print(f"Best Fitness (Cost): {self.global_best_cost:.2f}")
        print(f"Travel={self.global_best_details.get('travel',0)}, "
            f"Late={self.global_best_details.get('late',0)}, Wait={self.global_best_details.get('wait',0)}")
        print(f"Best tour: {self.global_best_tour}")
        return runtime, self.global_best_cost, progress