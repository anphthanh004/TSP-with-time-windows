import os
import numpy as np
from .gp_structures import Individual, Problem

"""
- Line 1: contains a positive integer `N` (1 ≤ N ≤ 1000)
- Line `i + 1` (i = 1, …, N): contains `e(i)`, `l(i)` and `d(i)`
- Line `i + N + 2` (i = 0, 1, …, N): contains the ith row of the matrix `t`
"""
def load_data(file_path):
    # data = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, 'r') as data:
        num_request = int(data.readline().strip())
        request = []
        time_matrix = []
        for _ in range(num_request):
            e, l, d = map(int, data.readline().strip().split())
            request.append((e, l, d))
        for _ in range(num_request + 1):
            row = list(map(int, data.readline().strip().split()))
            time_matrix.append(row)   
    return num_request, request, time_matrix

def load_output(input_path, output_path):
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"The file {output_path} does not exist.")
    input = load_data(input_path)
    problem = Problem(num_request=input[0],
                  request=input[1], 
                  time_matrix=np.array(input[2]))
    # problem = Problem()
    with open(output_path, 'r') as data:
        num_request = int(data.readline().strip())
        route = list(map(int, data.readline().strip().split()))
        indi = Individual(problem)
        indi.route = route
    return indi