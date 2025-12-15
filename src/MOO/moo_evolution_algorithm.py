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



def run_nsga2_1(problem, pop_size, c_rate, m_rate, generations, maximum_loop, **kwargs):
    print("Starting NSGA-II for Multi-Objectiv Optimization...")
    
    gen_type = kwargs.get('gen_type', 'greedy')
    greedy_rate = kwargs.get('greedy_rate', 0.5)
    search_size = kwargs.get('search_size', 2)
    params ={
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
    
    # sắp xếp các cá thể vào từng front
    current_fronts = fast_non_dominated_sorting(pop)
    for front in current_fronts:
        # tính khoảng cách quy tụ cho từng cá thể trong từng front
        crowding_distance_assignment(front)
    
    # best_travel_time = min(current_fronts[0], key = lambda x: sum(x.route_computing[0])).route_computing[0]
    best_travel_time = min(current_fronts[0], key = lambda x: x.fitness[0]).fitness[0]
    best_wait = min(current_fronts[0], key = lambda x: x.route_computing[7]).route_computing[7]
    best_lateness = min(current_fronts[0], key = lambda x: x.route_computing[5]).route_computing[5]
    
    # print (f"Initial fitness:  [travel_time: {sum(last_pop_best_indi.route_computing[0])}, total_wait: {last_pop_best_indi.route_computing[7]}, total_late: {last_pop_best_indi.route_computing[5]}")
    print (f"Initial fitness:  [best travel_time: {best_travel_time}, total_wait: {best_wait}, total_late: {best_lateness}")

    current_pop = pop
    # current_fronts = []
    # loop_not_improve = 0
    
    initial_m_rate = m_rate
    
    for i in range(generations):
        progress_pct = i / generations
        # setting mutation rate
        if progress_pct < 0.4:
            current_m_rate = initial_m_rate * (1 - 0.7 * progress_pct)
            params['mmethod'] = 'inversion' # Mạnh nhất (đảo ngược đoạn)
        elif progress_pct < 0.8:
            current_m_rate = initial_m_rate * (1 - 0.7 * progress_pct)
            params['mmethod'] = 'scramble' # Trung bình (xáo trộn trong đoạn)
        elif progress_pct < 0.95:
            current_m_rate = 0.5
            params['mmethod'] = 'swap'  # Nhẹ nhất (đổi chỗ hai request)
        else:
            current_m_rate = 0.1
        
        params['tourn_s_parameter'] = 2 + int(progress_pct * 2)
        
        # ----------------------------------------
        # Tạo thế hệ mới với params đã cập nhật
        # ----------------------------------------
        offspring = []
        while len(offspring) < pop_size:
            # Chọn lọc bằng Binary Tournament của NSGA-II
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
        cal_moo_fitness(offspring)
        # gộp hai quần thể
        combined_pop = offspring + [ind.copy() for ind in current_pop]
        
        cal_moo_fitness(combined_pop)
        # Chọn lọc các cá thể tốt nhất cho quần thể mới
        current_pop = nsga2_sv_selection(combined_pop, pop_size)
        
        current_fronts = fast_non_dominated_sorting(current_pop)
        # crowding_distance_assignment(current_pop)
        
        # pareto_front = fronts[0]
        if i % 20 == 0 or i == generations - 1:
            pareto_front = current_fronts[0]
            minimum_travel_time = min(pareto_front, key=lambda x: x.fitness[0]).fitness[0]
            # minimum_number_of_late_arrivals =   min(pareto_front, key=lambda x: x.fitness[1][0]).fitness[1][0]
            # minimum_total_late_arrival_time = min(pareto_front, key=lambda x: x.fitness[1][1]).fitness[1][1]
            minimum_violation = min(pareto_front, key=lambda x: x.fitness[1]).fitness[1]
            print(f'Gen: {i} | Pareto size: {len(pareto_front)}| Minimum travel time: {minimum_travel_time}| Minimum violation: {minimum_violation}')
        
        # if not dominate(current_fronts[0][0], last_pop_best_indi):
        #     loop_not_improve += 1
        # else:
        #     last_pop_best_indi = current_fronts[0][0]
        #     loop_not_improve = 0
            
        # if loop_not_improve == maximum_loop:
        #     print(f"Stop at gen {i} due to stagnation")
        #     break
        
    final_pareto = current_fronts[0]
    
    # f1_list = [ind.f1 for ind in final_pareto]
    f1_list = [ind.fitness[0] for ind in final_pareto]
    # f2_list = [ind.f2[0] for ind in final_pareto]
    f2_list = [ind.fitness[1] for ind in final_pareto]
    
    plt.figure(figsize=(10,6))
    plt.scatter(f1_list, f2_list, c='red', label='Pareto Front')
    plt.xlabel('Objective 1: Total travel time')
    plt.ylabel('Objective 2: Violation')
    plt.title('Pareto Front obtained by NSGA-II')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return current_pop


def run_nsga2(problem, pop_size, c_rate, m_rate, generations, maximum_loop, **kwargs):
    print("Starting NSGA-II for Multi-Objective Optimization...")
    
    # --- 1. Cấu hình tham số ---
    gen_type = kwargs.get('gen_type', 'greedy')
    greedy_rate = kwargs.get('greedy_rate', 0.5)
    search_size = kwargs.get('search_size', 2)
    params = {
        "cmethod": kwargs.get('cmethod', 'ox'),
        "mmethod": kwargs.get('mmethod', 'inversion'),
        "tourn_s_parameter": kwargs.get('tourn_s_parameter', 2),
    }
    
    # --- 2. Khởi tạo quần thể ban đầu ---
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
    
    # Tính toán fitness và phân cấp Front
    cal_moo_fitness(pop)
    current_fronts = fast_non_dominated_sorting(pop)
    for front in current_fronts:
        crowding_distance_assignment(front)
    
    # --- 3. Khởi tạo biến lưu trữ Progress (để vẽ biểu đồ) ---
    progress = []
    
    # Lấy giá trị tốt nhất của Objective 1 (Travel Time) tại thế hệ đầu tiên
    first_pareto_front = current_fronts[0]
    initial_best_obj1 = min(first_pareto_front, key=lambda x: x.fitness[0]).fitness[0]
    progress.append(initial_best_obj1)

    print(f"Initial fitness (Best Obj 1): {initial_best_obj1}")

    current_pop = pop
    initial_m_rate = m_rate
    
    # --- 4. Vòng lặp tiến hóa ---
    for i in range(generations):
        progress_pct = i / generations
        
        # --- A. Điều chỉnh tham số thích nghi (Adaptive Parameters) ---
        if progress_pct < 0.4:
            current_m_rate = initial_m_rate * (1 - 0.7 * progress_pct)
            params['mmethod'] = 'inversion' # Mạnh nhất (đảo ngược đoạn)
        elif progress_pct < 0.8:
            current_m_rate = initial_m_rate * (1 - 0.7 * progress_pct)
            params['mmethod'] = 'scramble' # Trung bình (xáo trộn trong đoạn)
        elif progress_pct < 0.95:
            current_m_rate = 0.5
            params['mmethod'] = 'swap'  # Nhẹ nhất (đổi chỗ hai request)
        else:
            current_m_rate = 0.1
        
        params['tourn_s_parameter'] = 2 + int(progress_pct * 2)
        
        # --- B. Lai ghép và Đột biến ---
        offspring = []
        while len(offspring) < pop_size:
            # Chọn lọc cha mẹ (Binary Tournament)
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
        # cal_moo_fitness(offspring)
        
        # --- C. Gộp và Chọn lọc sinh tồn (Survivor Selection) ---
        combined_pop = offspring + [ind.copy() for ind in current_pop]
        cal_moo_fitness(combined_pop)
        
        # Chọn ra pop_size cá thể tốt nhất dựa trên Rank và Crowding Distance
        current_pop = nsga2_sv_selection(combined_pop, pop_size)
        
        # Phân lớp lại quần thể mới để lưu progress
        current_fronts = fast_non_dominated_sorting(current_pop)
        # (Crowding distance assignment đã được gọi ngầm trong nsga2_sv_selection hoặc cần gọi lại nếu cần dùng tiếp)
        
        # --- D. Cập nhật Progress ---
        pareto_front = current_fronts[0]
        # Tìm cá thể có Travel Time (Obj 1) tốt nhất trong Pareto Front hiện tại
        best_obj1_gen = min(pareto_front, key=lambda x: x.fitness[0]).fitness[0]
        progress.append(best_obj1_gen)
        
        # --- E. Logging ---
        if i % 20 == 0 or i == generations - 1:
            minimum_violation = min(pareto_front, key=lambda x: x.fitness[1]).fitness[1]
            print(f'Gen: {i} | Pareto size: {len(pareto_front)} | Min Travel Time: {best_obj1_gen} | Min Violation: {minimum_violation}')
        
    # --- 5. Kết thúc ---
    final_pareto = current_fronts[0]
    

    return current_pop, progress