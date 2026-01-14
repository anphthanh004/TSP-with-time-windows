import math
import random
import numpy as np
import matplotlib.pyplot as plt
import copy


class Problem:
    def __init__(self, cities):
        self.num_cities = len(cities)
        self.coords = cities
        self.dist_matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    self.dist_matrix[i][j] = math.sqrt(
                        (cities[i][1] - cities[j][1])**2 + 
                        (cities[i][2] - cities[j][2])**2
                    )

class Individual:
    def __init__(self, problem, route=None):
        self.problem = problem
        self.route = route if route else []
        self.fitness = 0.0
        self.distance = 0.0

    def copy(self):
        new_ind = Individual(self.problem, self.route[:])
        new_ind.fitness = self.fitness
        new_ind.distance = self.distance
        return new_ind

    def cal_fitness(self):
        dist = 0
        n = len(self.route)
        mat = self.problem.dist_matrix
        for i in range(n - 1):
            dist += mat[self.route[i]][self.route[i+1]]
        dist += mat[self.route[-1]][self.route[0]]
        
        self.distance = dist
        self.fitness = 1.0 / dist 
        return self.distance


def load_data(filename):
    cities = [] 
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('[') or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    cities.append([int(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError: continue
    return cities

# --- Greedy Initialization (Nearest Neighbor) ---
def gen_greedy_individual(problem, start_node=None):
    n = problem.num_cities
    if start_node is None:
        start_node = random.randint(0, n-1)
    
    unvisited = set(range(n))
    unvisited.remove(start_node)
    route = [start_node]
    current = start_node
    
    while unvisited:
        nearest = min(unvisited, key=lambda x: problem.dist_matrix[current][x])
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest
        
    ind = Individual(problem, route)
    ind.cal_fitness()
    return ind

def gen_population(problem, pop_size, greedy_rate=0.2):
    pop = []
    num_greedy = int(pop_size * greedy_rate)
    for _ in range(num_greedy):
        start_node = random.randint(0, problem.num_cities - 1)
        pop.append(gen_greedy_individual(problem, start_node))
    while len(pop) < pop_size:
        perm = list(range(problem.num_cities))
        random.shuffle(perm)
        ind = Individual(problem, perm)
        ind.cal_fitness()
        pop.append(ind)
    return pop

# --- Selection (Tournament) ---
def tournament_selection(pop, k=4):
    candidates = random.sample(pop, k)
    best = max(candidates, key=lambda x: x.fitness)
    return best

# --- Crossover (Ordered Crossover - OX) ---
def ordered_crossover(parent1, parent2):
    n = len(parent1.route)
    start, end = sorted(random.sample(range(n), 2))
    
    child_route = [-1] * n
    child_route[start:end+1] = parent1.route[start:end+1]
    
    current_pos = (end + 1) % n
    for gene in parent2.route: #
        if gene not in child_route:
            child_route[current_pos] = gene
            current_pos = (current_pos + 1) % n
            
    return Individual(parent1.problem, child_route)

# --- Mutations (Adaptive) ---
def inversion_mutation(ind):
    n = len(ind.route)
    i, j = sorted(random.sample(range(n), 2))
    ind.route[i:j+1] = ind.route[i:j+1][::-1] 

def swap_mutation(ind):
    n = len(ind.route)
    i, j = random.sample(range(n), 2)
    ind.route[i], ind.route[j] = ind.route[j], ind.route[i]

# 3. LOCAL SEARCH
# Sử dụng 2-Opt Best Improvement 

def local_search_2opt(ind):
    route = ind.route
    problem = ind.problem
    n = len(route)
    mat = problem.dist_matrix
    improved = True
    best_dist = ind.distance
    
    count = 0 
    max_steps = 500 

    while improved and count < max_steps:
        improved = False
        count += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0: continue 
                
                
                node_i = route[i]
                node_i_next = route[i+1]
                node_j = route[j]
                node_j_next = route[(j+1)%n]
                
                current_len = mat[node_i][node_i_next] + mat[node_j][node_j_next]
                new_len = mat[node_i][node_j] + mat[node_i_next][node_j_next]
                
                if new_len < current_len - 1e-6:

                    route[i+1:j+1] = route[i+1:j+1][::-1]
                    best_dist -= (current_len - new_len)
                    improved = True

            
    ind.distance = best_dist
    ind.fitness = 1.0 / best_dist
    return ind


# 4. MAIN EVOLUTION

def run_ga_optimized():
    FILENAME = "tsp200_1.txt"
    POP_SIZE = 100
    GENERATIONS = 300 
    ELITE_SIZE = 5
    MUTATION_RATE_START = 0.5
    
    raw_cities = load_data(FILENAME)
    if not raw_cities: 
        print("Lỗi file input"); return
    
    cities_coords = [[0, c[1], c[2]] for c in raw_cities] 
    
    problem = Problem(cities_coords)
    print(f"Đã load {problem.num_cities} thành phố.")

    print("Đang khởi tạo quần thể (Hybrid: Random + Greedy)...")
    pop = gen_population(problem, POP_SIZE, greedy_rate=0.1)
    
    best_global = min(pop, key=lambda x: x.distance)
    print(f"Best ban đầu (Greedy Init): {best_global.distance:.2f}")
    
    progress = []
    
    for gen in range(GENERATIONS):
        progress_pct = gen / GENERATIONS
        

        current_m_rate = MUTATION_RATE_START * (1 - 0.8 * progress_pct)
        
        next_pop = []
        
        # Elitism: Giữ lại top cá thể tốt nhất
        pop.sort(key=lambda x: x.distance)
        next_pop.extend([ind.copy() for ind in pop[:ELITE_SIZE]])
        
        while len(next_pop) < POP_SIZE:
            # Selection
            p1 = tournament_selection(pop, k=5)
            p2 = tournament_selection(pop, k=5)
            
            # Crossover (OX)
            child = ordered_crossover(p1, p2)
            
            # Mutation (Adaptive Method)
            if random.random() < current_m_rate:
                if progress_pct < 0.7:
                    inversion_mutation(child)
                else:
                    swap_mutation(child)
            
            if random.random() < 0.05 or (progress_pct > 0.9 and random.random() < 0.2):
                child.cal_fitness() 
                child = local_search_2opt(child)
            
            child.cal_fitness()
            next_pop.append(child)
            
        pop = next_pop
        
        current_best = min(pop, key=lambda x: x.distance)
        if current_best.distance < best_global.distance:
            best_global = current_best.copy()
            
        # progress.append(best_global.distance)
        progress.append(best_global)
        
        if gen % 20 == 0:
            print(f"Gen {gen}: Best Dist = {best_global.distance:.2f} | M_Rate={current_m_rate:.2f}")

    print("\n-------------------------------------------")
    print(f"KẾT QUẢ TỐI ƯU CUỐI CÙNG: {best_global.distance:.2f}")
    print(min(progress,key=lambda x: x.distance).route)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot([x.distance for x in progress])
    plt.title('Mức độ hội tụ (Fitness)')
    plt.xlabel('Thế hệ')
    plt.ylabel('Khoảng cách (Distance)')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    x = [problem.coords[i][1] for i in best_global.route]
    y = [problem.coords[i][2] for i in best_global.route]
    x.append(x[0])
    y.append(y[0])
    
    plt.plot(x, y, 'o-', c='r', markersize=4, linewidth=1)
    plt.title(f'Lộ trình tối ưu (Dist: {best_global.distance:.2f})')
    plt.show()

if __name__ == "__main__":
    run_ga_optimized()