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
    type: str = 'GA'        
    penalty: float = 1000.0 

class Individual:
    def __init__(self, problem=None):
        """
        Individual chỉ tham chiếu (reference) tới Problem => quan hệ Association (liên kết)
        Individual ---(Association)---> Problem
        """   
        self.route = []
        self.problem = problem
        self.fitness = None
        # self.valid = []
        # self.late = []
        # self.wait = []
        # self.total_service_time = float('inf')
        
        # ---Local Search---
        # e, l dùng cho feasibility check
        # e : earlieset_arrival[i] thời điểm sớm nhất có thể đến vị trí i, l là latest_arrival[i] thời điểm trễ nhất vẫn có thể đến
        # self.earlieset_arrival = []
        # self.latest_arrival = []
        self.route_computing = None
        
    # def copy(self):
    #     return copy.deepcopy(self)
    def copy(self):
        """
        Tạo bản sao cá thể thủ công để tránh deepcopy toàn bộ object Problem
        giúp tiết kiệm bộ nhớ khi dữ liệu lớn.
        """
        # Tạo cá thể mới, truyền tham chiếu problem cũ vào (KHÔNG copy problem)
        new_ind = Individual(self.problem)
        # Copy lộ trình (quan trọng nhất)
        # Sử dụng slicing [:] để tạo shallow copy của list, nhanh hơn deepcopy
        new_ind.route = self.route[:]
        
        return new_ind
    
    # def compute_forward_backward_times(route, problem):
    #     n = len(problem.num_request)
    #     e = [0] * n
    #     l = [0] * n
    #     # Forward pass
    #     current_time = 0
    #     prev = 0
    #     for i in range(n):
    #         node = route[i]
    #         travel = problem.time_matrix[prev][node]
    #         earliest, latest, duration = problem.request[node-1]
            
    #         arrival = current_time + travel
    #         if arrival < earliest:
    #             arrival = earliest
            
    #         e[i] = arrival
    #         current_time = arrival + duration
    #         prev = node
            
    #     # Backward pass
    #     current_time = 0
    #     next_node = 0
    #     for i in reversed(range(n)):
    #         node = route[i]
    #         travel = problem.time_matrix[node][next_node]
    #         earliest, latest, duration = problem.request[node-1]
            
    #         latest_arrival = latest if i == n - 1 else l[i+1] - travel - duration
            
    #         l[i] = latest_arrival
    #         next_node = node
        
    #     return e, l
            
        
        # current_time = 0
    def compute_route_forward(self, route, problem):
        n = len(route)
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
            arrival = current_time + travel
            # waiting allowed: arrive earlier -> wait
            if arrival < e_i:
                arrival = e_i
            # lateness allowed: measure how much late
            late = max(0.0, arrival - l_i)
            wait.append(max(0.0, e_i - arrival))
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
        return arrivals, departures, total_time, lateness, total_lateness, wait, total_wait
    # def calObjective(self, problem):
    #     #---------------------
    #     # Rest mỗi khi gọi lại (có thể lúc này nó thuộc quần thể mới)
    #     self.late = []
    #     self.wait = []
    #     self.total_service_time = 0
    #     # self.objective = float('inf')
    #     self.objective = None
        
        
    #     if self.problem.type == 'MOO': 
    #         self.distance = None
    #         self.rank = None
    #     #----------------------
    #     current_time = 0
    #     current_location = 0
    #     for location in self.route:
    #         earliest, latest, duration = problem.request[location-1] # e_i, l_i, d_i
    #         current_time += problem.time_matrix[current_location][location]
    #         if current_time < earliest:
    #             wait_time = earliest-current_time
    #             current_time = earliest
    #             # self.valid.append(True)
    #             self.late.append(0)
    #             self.wait.append(wait_time)
    #         elif earliest <= current_time <= latest:
    #             # self.valid.append(True)
    #             self.late.append(0)
    #             self.wait.append(0)
    #         elif current_time > latest:
    #             # self.valid.append(False)
    #             self.late.append(current_time - latest)
    #             self.wait.append(0)
                
    #         current_time += duration # service time
    #         current_location = location
        
    #     current_time += problem.time_matrix[current_location][0] # return to depot
    #     self.total_service_time = current_time
    #     if self.problem.type == 'GA':
    #         self.objective = current_time + problem.penalty * sum(self.late)
    #         return self.objective
    #     if self.problem.type == 'MOO':
    #         nums_of_arrivals_late = sum(1 for x in self.late if x != 0)
    #         total_time_late = sum(self.late) 
    #         # thời gian hoàn thành, số lần đến muộn, tổng thời gian đến muộn
    #         self.objective = (current_time, nums_of_arrivals_late, total_time_late)
    def calObjective(self, problem):
        #---------------------
        # Rest mỗi khi gọi lại (có thể lúc này nó thuộc quần thể mới)
        self.objective = None
        self.route_computing = None
        
        
        if self.problem.type == 'MOO': 
            self.distance = None
            self.rank = None
        #----------------------
        
        # 0-arrivals, 1-departures, 2-total_time, 3-lateness, 4-total_lateness, 5-wait, 6-total_wait
        self.route_computing = self.compute_route_forward(self.route, problem)
        if self.problem.type == 'GA':
            total_time = self.route_computing[2]
            total_lateness = self.route_computing[4]
            self.objective = total_time + problem.penalty * total_lateness
            return self.objective
        if self.problem.type == 'MOO':
            total_time = self.route_computing[2]
            lateness = self.route_computing[3]
            total_lateness = self.route_computing[4]
            nums_of_late_arrivals = sum(1 for x in lateness if x != 0.0)
            # thời gian hoàn thành, số lần đến muộn, tổng thời gian đến muộn
            self.objective = (total_time, nums_of_late_arrivals, total_lateness)
        