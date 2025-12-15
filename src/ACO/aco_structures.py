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

        prob_list = [] # lưu các node và value của node 
        roulette_wheel = 0.0 # lưu tổng value
        
        edges = self.edges
        problem = self.problem
        
        for node in unvisited_nodes:
            # Công thức: [tau]^alpha * [eta]^beta
            # tau = self.edges[current_node][node].pheromone (pheromone)
            # eta (heuristic) # Heuristic tính dựa theo distance và wait và cũng chịu ảnh hưởng bởi mức độ khẩn cấp
            travel_time = edges[current_node][node].weight
            arrival_time = current_time + travel_time
            e, l, d = problem.request[node-1]
            wait_time = max(0.0, e-arrival_time)
            delay = travel_time + wait_time
            depature = max(arrival_time, e) + d
            
            urgency = l - arrival_time
            if urgency < 0:
                eta = 0.0001 
            else:
                # ưu tiên delay nhỏ và ưu tiên node sắp đóng cửa
                eta = 1.0 / (delay + 0.1 * urgency + 1.0)
            # lấy pheromone
            tau = edges[current_node][node].pheromone
            value = pow(tau, self.alpha) * pow(eta, self.beta)
            roulette_wheel += value
            # prob_list.append((node, value))   
            prob_list.append((node, value, depature))    
        
        if roulette_wheel == 0:
            return random.choice(unvisited_nodes)
        
        random_value = random.uniform(0.0, roulette_wheel)
        wheel_position = 0.0
        
        for node, value, depature in prob_list:
            wheel_position += value
            if wheel_position > random_value:
                return node, depature
    
    def find_tour(self):

        tour = []
        unvisited = list(range(1, self.problem.num_request + 1))
        current_node = 0
        current_time = 0.0
        
        while unvisited:
            next_node, depature = self._select_node(current_node, unvisited, current_time)
            tour.append(next_node)
            unvisited.remove(next_node)
            current_time = depature
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
        
        tm = problem.time_matrix
        rq = problem.request
        
        for idx in range(n):
            node = route[idx]
            e_i, l_i, d_i = rq[node-1]
            
            travel = tm[prev][node]
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
        return_travel = tm[prev][0]
        current_time += return_travel
        total_travel += return_travel
        
        self.travel_time = total_travel
        self.total_lateness = total_lateness
        self.total_wait = total_wait
        
        penalty = problem.penalty
        self.total_cost = total_travel + total_wait * penalty[0] + total_lateness * penalty[1]
        
        return self.total_cost
    
                        
            
        