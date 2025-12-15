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
        self.mode = mode                                # chọn ACS hoặc Elitist hoặc MaxMin
        self.colony_size = colony_size                  # kích thước đàn kiến
        self.elitist_weight = elitist_weight            # dùng cho mode='Elitist': trọng số thêm pheromone cho cạnh thuộc đường đi tốt nhất 
                                                                # (để nhân với pheromone do kiến có đường đi tốt nhất để lại)
        self.min_scaling_factor = min_scaling_factor    # dùng cho mode='MinMax': min bằng bao nhiêu % của max
        self.rho = rho                                  # hệ số bay hơi
        self.pheromone_deposit_weight = pheromone_deposit_weight
        self.steps = steps                              # thanh tiến trình tqdm
        
        self.num_nodes = problem.num_request + 1
        
        self.edges = [[None]* self.num_nodes for _ in range(self.num_nodes)]
        n = self.num_nodes
        tm = problem.time_matrix
        for i in range(n):
            for j in range(n):
                weight = tm [i][j]
                # chỗ này tức là hai chiều i -> j coi là cạnh khác nhau (vì đi phục vụ request A rồi đến B khác với đi phục vụ request B rồi đến A)
                # do đó với logic này, pheromone trên i -> j cũng khác j -> i (~Directed Graph)
                self.edges[i][j] = Edge(i, j, weight, initial_pheromone) # Asymmetric
        
        self.ants = [Ant(alpha, beta, self.edges, self.problem) for _ in range(self.colony_size)]
        
        self.global_best_cost = float('inf')
        self.global_corresponding_tour = None
        self.global_corresponding_details = {}
    
    # cập nhật pheromone: pheromone_(i,j) <- (1-rho)*pheromone_(i,j) + delta(k)_(i,j) 
                # trong đó: rho là tốc độ bay hơi, delta(k)_(i,j) là pheromone để mà con kiến k để lại trên cạnh (i,j) thuộc đường đi của nó
    
    # Thêm pheromone   (delta(k)_(i,j) trong đó tour là để xác định(i,j) mà ngoài con kiến này thì có thể con kiến khác cũng để lại pheromone trên(i,j))
                        # cost là để xác định cho chính xác con kiến này, lượng để lại là additional_pheromone được xác định trong hàm dưới
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
                self._add_pheromone(ant.tour, cost) # thêm pheromone trên cạnh (i,j)
                if cost < self.global_best_cost:
                    self.global_best_cost = cost
                    self.global_corresponding_tour = ant.tour
                    self.global_corresponding_details={"travel": ant.travel_time, "late": ant.total_lateness, "wait": ant.total_wait}
            
            # Sau khi tất cả kiến hoàn thành chặng đường, pheromone trên cả đồ thị sẽ bị bay hơi
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
                    self.global_best_cost = cost
                    self.global_corresponding_tour =ant.tour
                    self.global_corresponding_details={"travel": ant.travel_time, "late": ant.total_lateness, "wait": ant.total_wait}
            # cập nhật pheromone: pheromone_(i,j) <- (1-rho)*pheromone_(i,j) + delta(k)_(i,j) 
            # lúc này pheromone trên cạnh (i,j) của lộ trình tốt nhất toàn cục được cập nhật thêm một lượng:
            #               delta(best)_(i,j) * elitist_weight (đánh hệ số)
            if self.global_corresponding_tour:
                self._add_pheromone(self.global_corresponding_tour, self.global_best_cost, weight=self.elitist_weight)
            
            # tiếp theo là bay hơi
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
                    iteration_best_cost = cost
                    iteration_corresponding_tour = ant.tour
                    iteration_corresponding_travel_time = ant.travel_time
                    iteration_total_lateness = ant.total_lateness
                    iteration_total_wait = ant.total_wait
            
            # Cập nhật Global Best
            if iteration_best_cost < self.global_best_cost:
                self.global_best_cost = iteration_best_cost
                self.global_corresponding_tour = iteration_corresponding_tour
                self.global_corresponding_details={"travel": iteration_corresponding_travel_time, "late": iteration_total_lateness, "wait": iteration_total_wait}

            # Cập nhật Pheromone
            if float(step + 1) / float(self.steps) <= 0.75:
                self._add_pheromone(iteration_corresponding_tour, iteration_best_cost)
                max_pheromone = self.pheromone_deposit_weight / iteration_best_cost if iteration_best_cost > 0 else 10.0
            else:
                self._add_pheromone(self.global_corresponding_tour, self.global_best_cost)
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
        print(f"Travel={self.global_corresponding_details.get('travel',0)}, "
            f"Late={self.global_corresponding_details.get('late',0)}, Wait={self.global_corresponding_details.get('wait',0)}")
        print(f"Best tour: {self.global_corresponding_tour}")
        return runtime, self.global_best_cost, progress