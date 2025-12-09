import numpy as np
import random
import operator
import matplotlib.pyplot as plt
import copy
from dataclasses import dataclass

# Problem chỉ chứa dữ liệu đề bài
@dataclass(frozen=True) 
class Problem:
    time_matrix: np.ndarray 
    request: list           
    num_request: int              
    penalty: tuple[float,float] = 2.0, 4.0 

class Individual:
    def __init__(self, problem=None):
        """
        Individual chỉ tham chiếu (reference) tới Problem => quan hệ Association (liên kết)
        Individual ---(Association)---> Problem
        """   
        self.route = []
        self.problem = problem
        self.fitness = None
        self.route_computing = None
        
    def copy(self):
        new_ind = Individual(self.problem)
        new_ind.route = self.route[:]
        return new_ind
        
    def compute_route_forward(self):
        route = self.route
        problem = self.problem
        n = len(route)
        travels = [0]*n
        arrivals = [0]*n
        departures = [0]*n
        lateness = [0]*n
        wait = [0]*n
        
        current_time = 0
        prev = 0
        for idx in range(n):
            node = route[idx]
            e_i, l_i, d_i = problem.request[node-1]
            travel = problem.time_matrix[prev][node]
            travels[idx] = travel 
            arrival = current_time + travel
            wait[idx]=max(0.0, e_i - arrival)
            # waiting allowed: arrive earlier -> wait
            if arrival < e_i:
                arrival = e_i
            # lateness allowed: measure how much late
            late = max(0.0, arrival - l_i)
            # wait.append(max(0.0, e_i - arrival))
            arrivals[idx] = arrival
            lateness[idx] = late
            departures[idx] = arrival + d_i
            current_time = departures[idx]
            prev = node

        # return to depot
        current_time += problem.time_matrix[prev][0]
        total_time = current_time
        total_lateness = sum(lateness)
        total_wait = sum(wait)
        return travels, arrivals, departures, total_time, lateness, total_lateness, wait, total_wait

    def calObjective(self):
        problem = self.problem
        #---------------------
        # Rest mỗi khi gọi lại (có thể lúc này nó thuộc quần thể mới)
        self.objective = None
        self.route_computing = None
        self.distance = None
        self.rank = None
        #----------------------
        
        self.route_computing = self.compute_route_forward(self)

        # 0-travels, 1-arrivals, 2-departures, 3-total_time, 4-lateness, 5-total_lateness, 6-wait, 7-total_wait
        self.route_computing = self.compute_route_forward()
        travel_time = self.route_computing[0]
        total_lateness = self.route_computing[5]
        total_wait = self.route_computing[7]
        # self.objective = sum(travel_time) + problem.penalty[0] * total_wait + problem.penalty[1] * total_lateness
        self.objective = (sum(travel_time), total_wait*problem.penalty[0] + total_lateness*problem.penalty[1])
        return self.objective
        

