import random
# import numpy
import matplotlib.pyplot as plt
from .moo_operators import cal_moo_fitness, \
                        nsga2_tourn_selection,nsga2_sv_selection,\
                        apply_mutation, perform_crossover
from .moo_initialization import gen_pop, gen_pop_fully_random, \
                            gen_pop_greedy1, gen_pop_greedy2,\
                            gen_pop_greedy3, gen_pop_greedy4
from .nsga2_algorithm import *


def run_nsga2(problem, pop_size, c_rate, m_rate, generations, maximum_loop, **kwargs):
    print("Starting NSGA-II for Multi-Objective Optimization...")

    gen_type = kwargs.get('gen_type', 'greedy')
    greedy_rate = kwargs.get('greedy_rate', 0.5)
    search_size = kwargs.get('search_size', 2)
    params = {
        "cmethod": kwargs.get('cmethod', 'ox'),
        "mmethod": kwargs.get('mmethod', 'inversion'),
        "tourn_s_parameter": kwargs.get('tourn_s_parameter', 2),
    }
    
    if gen_type == 'random':
        pop = gen_pop_fully_random(problem, pop_size)
    elif gen_type == 'greedy':
        pop = gen_pop(problem, greedy_rate, search_size, pop_size)
    elif gen_type == 'greedy1':
        pop = gen_pop_greedy1(problem, greedy_rate, search_size, pop_size)
    elif gen_type == 'greedy2':
        pop = gen_pop_greedy2(problem, greedy_rate, search_size, pop_size)
    elif gen_type == 'greedy3':
        pop = gen_pop_greedy3(problem, greedy_rate, search_size, pop_size)
    elif gen_type == 'greedy4':
        pop = gen_pop_greedy4(problem, greedy_rate, search_size, pop_size)    
    

    cal_moo_fitness(pop)
    current_fronts = fast_non_dominated_sorting(pop)
    for front in current_fronts:
        crowding_distance_assignment(front)
    

    progress = []
    
    first_pareto_front = current_fronts[0]
    initial_best_obj1 = min(first_pareto_front, key=lambda x: x.fitness[0]).fitness[0]
    progress.append(initial_best_obj1)

    print(f"Initial fitness (Best Obj 1): {initial_best_obj1}")

    current_pop = pop
    # initial_m_rate = m_rate
    current_m_rate = m_rate

    for i in range(generations):
        progress_pct = i / generations
        
        if progress_pct < 0.4:
            # current_m_rate = initial_m_rate * (1 - 0.7 * progress_pct)
            params['mmethod'] = 'inversion' # Mạnh nhất (đảo ngược đoạn)
        elif progress_pct < 0.8:
            # current_m_rate = initial_m_rate * (1 - 0.7 * progress_pct)
            params['mmethod'] = 'scramble' # Trung bình (xáo trộn trong đoạn)
        # elif progress_pct < 0.95:
        else:
            # current_m_rate = 0.5
            params['mmethod'] = 'swap'  # Nhẹ nhất (đổi chỗ hai request)
        # else:
        #     current_m_rate = 0.1
        
        # params['tourn_s_parameter'] = 2 + int(progress_pct * 2)
        # params['tourn_s_parameter'] = 2
        

        offspring = []
        while len(offspring) < pop_size:
            p1 = nsga2_tourn_selection(pop, params['tourn_s_parameter'])
            p2 = nsga2_tourn_selection(pop, params['tourn_s_parameter'])

            # Lai ghép
            if random.random() < c_rate:
                c1, c2 = perform_crossover(p1, p2, params['cmethod'])
            else:
                c1, c2 = p1.copy(), p2.copy()
            
            # Đột biến
            if random.random() < current_m_rate:
                c1 = apply_mutation(c1, params['mmethod'])
            if random.random() < current_m_rate:
                c2 = apply_mutation(c2, params['mmethod'])
            
            offspring.extend([c1, c2])
        
        offspring = offspring[:pop_size]

        combined_pop = offspring + [ind.copy() for ind in current_pop]
        cal_moo_fitness(combined_pop)

        current_pop = nsga2_sv_selection(combined_pop, pop_size)
        current_fronts = fast_non_dominated_sorting(current_pop)
        pareto_front = current_fronts[0]
        # Tìm cá thể có Travel Time (Obj 1) tốt nhất trong Pareto Front hiện tại
        best_obj1_gen = min(pareto_front, key=lambda x: x.fitness[0]).fitness[0]
        progress.append(best_obj1_gen)
        

        if i % 20 == 0 or i == generations - 1:
            minimum_violation = min(pareto_front, key=lambda x: x.fitness[1]).fitness[1]
            print(f'Gen: {i} | Pareto size: {len(pareto_front)} | Min Travel Time: {best_obj1_gen} | Min Violation: {minimum_violation}')
        
    final_pareto = current_fronts[0]
    

    return current_pop, progress