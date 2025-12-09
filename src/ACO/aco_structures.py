import numpy as np
import random
import matplotlib.pyplot as plt
import copy
from dataclasses import dataclass

# Problem chỉ chứa dữ liệu đề bài
@dataclass(frozen=True) 
class Problem:
    time_matrix: np.ndarray 
    request: list           
    num_request: int           
    penalty: tuple[float, float] =(10.0, 1000.0)

class Edge:
    def __init__(self, a, b, weight, initial_pheromone):
        self.a = a
        self.b = b
        # weight là thời gian di chuyển cơ bản (để tính heuristic)
        if weight == 0:
            weight = 1e-10
        self.weight = weight
        self.pheromone = initial_pheromone

class Ant:
    def __init__(self, alpha=None, beta=None, edges=None, problem=None):
        self.problem = problem
        self.alpha = alpha
        self.beta = beta
        self.edges = edges
        self.tour = []
        self.total_cost = 0.0
        self.travel_time = 0
        self.total_lateness = 0.0
        self.total_wait = 0.0
    
    def _select_node(self, current_node, unvisited_nodes, current_time):
        # Tính mẫu số chung cho Roulette wheel
        roulette_wheel = 0.0
        heuristic_total = 0.0
        # Tính xác suất tích lũy
        prob_list = []
        
        # # tính tổng heuristic (eta) của các node chưa thăm
        # for node in unvisited_nodes:
        #     heuristic_total += (1.0/self.edges[current_node][node].weight)
            
        for node in unvisited_nodes:
            # Công thức: [tau]^alpha * [eta]^beta
            # tau = self.edges[current_node][node].pheromone
            # eta = (1.0 / self.edges[current_node][node].weight)  # Heuristic nghịch đảo thời gian di chuyển
            dist = self.edges[current_node][node].weight
            arrival_time = current_time + dist
            
            e, l, d = self.problem.request[node-1]
            wait_time = max(0.0, e-arrival_time)
            urgency = l - arrival_time
            delay = dist + wait_time
            
            if urgency < 0:
                eta = 0.0001 
            else:
                # ưu tiên delay nhỏ và ưu tiên node sắp đóng cửa
                eta = 1.0 / (delay + 0.1 * urgency + 1.0)
            # lấy pheromone
            tau = self.edges[current_node][node].pheromone
            value = pow(tau, self.alpha) * pow(eta, self.beta)
            roulette_wheel += value
            prob_list.append((node, value))       
        
        if roulette_wheel == 0:
            return random.choice(unvisited_nodes)
        random_value = random.uniform(0.0, roulette_wheel)
        wheel_position = 0.0
        
        for node, value in prob_list:
            wheel_position += value
            if wheel_position > random_value:
                return node
        
        return unvisited_nodes[-1] # Fallback
    
    def find_tour(self):
        current_node = 0
        tour = []
        unvisited = list(range(1, self.problem.num_request + 1))
        
        current_time = 0.0
        
        while unvisited:
            next_node = self._select_node(current_node, unvisited, current_time)
            tour.append(next_node)
            unvisited.remove(next_node)
            # Cập nhật thời gian hiện tại sau khi lấy được node
            dist = self.problem.time_matrix[current_node][next_node]
            arrival = current_time + dist
            e, l, d = self.problem.request[next_node - 1]
            start_service = max(arrival, e)
            current_time = start_service + d
            current_node = next_node
        self.tour = tour
        
        return tour
    
    def calculate_cost(self):
        """
        cost = total_travel_time + phạt chờ + phạt trễ
        """
        route = self.tour
        problem = self.problem
        n = len(route)
        
        current_time = 0
        prev = 0
        
        total_travel = 0
        total_lateness = 0
        total_wait = 0
        
        for idx in range(n):
            node = route[idx]
            e_i, l_i, d_i = problem.request[node-1]
            
            travel = problem.time_matrix[prev][node]
            total_travel += travel
            
            arrival = current_time + travel
            
            wait = max(0.0, e_i - arrival)
            total_wait += wait
            
            start_service = max(arrival, e_i)
            late = max(0.0, start_service - l_i)
            total_lateness += late
            
            current_time = start_service + d_i
            prev = node
            
        # Quay về depot
        return_travel = problem.time_matrix[prev][0]
        current_time += return_travel
        total_travel += return_travel
        
        self.travel_time = total_travel
        self.total_lateness = total_lateness
        self.total_wait = total_wait
        
        penalty = problem.penalty
        self.total_cost = total_travel + total_wait * penalty[0] + total_lateness * penalty[1]
        
        return self.total_cost
    
                        
            
        