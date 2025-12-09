import random
# import numpy
import matplotlib.pyplot as plt
# Import các module GP
from .gp_structures import Individual
from .gp_operators import cal_std_fitness, gp_crossover, gp_mutation
from .gp_initialization import gen_pop


def run_gp_algorithm(problem, pop_size, c_rate, m_rate, generations, maximum_loop, **kwargs):
    print("Starting Genetic Programming...")
    
    # 1. Khởi tạo quần thể GP
    current_pop = gen_pop(problem, pop_size, max_depth=6)
    
    # 2. Đánh giá fitness ban đầu
    cal_std_fitness(current_pop) 
    # best_ind = min(current_pop, key=lambda x: x.fitness)
    current_best = min(current_pop, key=lambda x: x.fitness)
    print(f"Initial GP fitness: {current_best.fitness}")
    
    loop_no_improve = 0
    last_fitness = current_best.fitness
    progress = [last_fitness]
    
    for gen in range(generations):
        next_pop = []
        
        # # Elitism: Giữ lại 1 cá thể tốt nhất
        # next_pop.append(best_ind.copy())
        
        while len(next_pop) < pop_size:
            # Selection: Tournament đơn giản
            candidates = random.sample(current_pop, 3)
            parent1 = min(candidates, key=lambda x: x.fitness)
            candidates = random.sample(current_pop, 3)
            parent2 = min(candidates, key=lambda x: x.fitness)
            
            # Crossover
            if random.random() < c_rate:
                child1, child2 = gp_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < m_rate:
                child1 = gp_mutation(child1)
            if random.random() < m_rate:
                child2 = gp_mutation(child2)
            
            # # Reset route cũ để cây mới sinh ra route mới
            # child1.route = []
            # child2.route = []
            
            next_pop.extend([child1, child2])
            
        # # Cắt tỉa về đúng pop_size
        # pop = next_pop[:pop_size]
        best = min(current_pop, key=lambda x: x.fitness).copy()
        cal_std_fitness([best])
        
        combined_pop = next_pop + [ind.copy() for ind in current_pop]
        cal_std_fitness(combined_pop)
        
        combined_pop.sort(key=lambda x: x.fitness)
        current_pop = combined_pop[:pop_size]
        
        if best.tree.to_string() not in [ind.tree.to_string() for ind in current_pop]:
            current_pop = combined_pop[:pop_size-1]
            current_pop.append(best)
        
        current_best = min(current_pop, key=lambda x: x.fitness)
        # In log định kỳ
        if gen % 10 == 0:
            print(f"Gen {gen}: Best Fitness = {current_best.fitness}")
            
        progress.append(current_best.fitness)
        
        # Kiểm tra điều kiện dừng
        if current_best.fitness < best.fitness:
            loop_no_improve = 0
        else:
            loop_no_improve += 1
            
        if loop_no_improve >= maximum_loop:
            print(f"Stopping due to stagnation at gen {gen}.")
            break
            
    # Vẽ biểu đồ
    plt.plot(progress)
    plt.title("GP Evolution Progress")
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.show()
    
    print("Best GP Route:", current_best.route)
    print("Best GP Rule:", current_best.tree.to_string())
    
    return current_pop