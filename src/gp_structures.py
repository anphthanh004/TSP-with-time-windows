import numpy as np
import random
import operator
import matplotlib.pyplot as plt
import copy
from dataclasses import dataclass
from .gp_simulation import simulate_tsptw

# Problem chỉ chứa dữ liệu đề bài
@dataclass(frozen=True) 
class Problem:
    time_matrix: np.ndarray 
    request: list           
    num_request: int        
    type: str = 'GP'        
    penalty: float = 1000.0 

# ------------------------------------------------
# 1. Tập hàm (Function Set) & Terminal Set
# ------------------------------------------------
def protected_div(a, b):
    return a / b if abs(b) > 0.001 else 1.0

FUNC_SET = ['add', 'sub', 'mul', 'div', 'min', 'max']

# Định nghĩa các Terminal cho TSPTW (Priority Rules)
# ST0: Travel Time (Distance)
# ST1: Ready Time (e_i)
# ST2: Due Date (l_i)
# ST3: Waiting Time (max(0, e_i - arrival))
# ST4: Slack Time (l_i - arrival) - Độ khẩn cấp
TERMINAL_SET = [('R', i) for i in range(5)]

# ------------------------------------------------
# 1. Cây và cá thể
# ------------------------------------------------
class NodeGP:
    def __init__(self, op=None, left=None, right=None, terminal=None, penalty=None):
        self.op = op
        self.left = left
        self.right = right
        self.terminal = terminal # ('R', index)
        self.penalty = penalty

    def is_terminal(self):
        return self.terminal is not None

    def deepcopy(self):
        return copy.deepcopy(self)
        
    def size(self):
        if self.is_terminal(): return 1
        return 1 + (self.left.size() if self.left else 0) + (self.right.size() if self.right else 0)

    def depth(self):
        if self.is_terminal(): return 1
        return 1 + max(self.left.depth() if self.left else 0, self.right.depth() if self.right else 0)

    def to_string(self):
        if self.terminal is not None:
            _, opt = self.terminal
            names = ['Dist', 'Ready', 'Due', 'Wait', 'Slack']
            return names[opt]
        return f"({self.op} {self.left.to_string()} {self.right.to_string()})"

    def evaluate(self, dist, ready, due, wait, slack):
        if self.terminal is not None:
            _, opt = self.terminal
            # maximize 
            if opt == 0: return -dist  
            if opt == 1: return -ready
            if opt == 2: return due
            if opt == 3: return -wait
            # if opt == 4: return slack if slack > 0 else slack * self.problem.penalty
            if opt == 4: return slack if slack > 0 else slack * self.penalty
            # return 0.0
            
        # Function nodes
        val_l = self.left.evaluate(dist, ready, due, wait, slack)
        val_r = self.right.evaluate(dist, ready, due, wait, slack)

        if self.op == 'add': return val_l + val_r
        if self.op == 'sub': return val_l - val_r
        if self.op == 'mul': return val_l * val_r
        if self.op == 'div': return protected_div(val_l, val_r)
        if self.op == 'min': return min(val_l, val_r)
        if self.op == 'max': return max(val_l, val_r)

class Individual:
    def __init__(self, problem=None, tree=None):
        """
        Individual chỉ tham chiếu (reference) tới Problem => quan hệ Association (liên kết)
        Individual ---(Association)---> Problem
        """   
        # self.route = []
        self.problem = problem
        self.tree = tree
        self.fitness = None
        self.route = []
        self.route_computing = None
        
    # def copy(self):
    #     return copy.deepcopy(self)
    def copy(self):
        """
        Tạo bản sao cá thể thủ công để tránh deepcopy toàn bộ object Problem
        giúp tiết kiệm bộ nhớ khi dữ liệu lớn.
        """
        new_ind = Individual(self.problem, self.tree)
        new_ind.route = self.route[:]
        
        return new_ind

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

    def calObjective(self, problem):
        #---------------------
        # Rest mỗi khi gọi lại (có thể lúc này nó thuộc quần thể mới)
        self.objective = None
        self.route_computing = None
        
        if self.problem.type == 'GP' and self.tree is not None:
            # self.tree.penalty = problem.penalty
            # Import bên trong hàm để tránh lỗi Circular Import với file gp_simulation
            simulate_tsptw(self, problem)
        
        # 0-arrivals, 1-departures, 2-total_time, 3-lateness, 4-total_lateness, 5-wait, 6-total_wait
        self.route_computing = self.compute_route_forward(self.route, problem)
        if self.problem.type == 'GP':
            total_time = self.route_computing[2]
            total_lateness = self.route_computing[4]
            self.objective = total_time + problem.penalty * total_lateness
            return self.objective

        