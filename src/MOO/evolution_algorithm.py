import random
# import numpy
import matplotlib.pyplot as plt
from .operators import calculate_fitness, calculate_mo_fitness, \
                        select_parents, apply_mutation, perform_crossover, apply_sv_selection
from .initialization import gen_pop, gen_pop_fully_random, \
                            gen_pop_greedy1, gen_pop_greedy2,\
                            gen_pop_greedy3, gen_pop_greedy4
from .nsga2_algorithm import *
# Import các module GP
from .gp_structures import Individual
from .gp_op import create_population, gp_crossover, gp_mutation

# from .local_search import local_search_softTW, local_search_softTW_best_improvement
# from .local_search import local_search_softTW_best_improvement, local_search_softTW_first_improvement
from .local_search_v2 import local_search_softTW_best_improvement, local_search_softTW_first_improvement
def create_next_population(pop, problem, c_rate, m_rate, **kwargs):
    POPSIZE = len(pop)
    children_list = []
    # lấy kiểu
    fmethod = kwargs.get('fmethod')
    smethod = kwargs.get('smethod')
    cmethod = kwargs.get('cmethod')
    mmethod = kwargs.get('mmethod')
    svmethod = kwargs.get('svmethod')
    
    imp_type = kwargs.get('imp_type', 'first_improvement')
    ls_rate = kwargs.get('ls_rate', 0.1)
    
    #1. Tạo con (crossover)
    while len(children_list) < POPSIZE:
        # lựa chọn cha mẹ
         # lấy tham số cho từng kiểu chọn cha mẹ
        tourn_s_size = kwargs.get('tourn_s_parameter')
        top_k = kwargs.get('ranking_s_parameter')
        # smethod {'random', 'roulette', 'tournament', 'ranking'}
        p1, p2 = select_parents(pop, smethod, tourn_s_parameter= tourn_s_size, ranking_parameter=top_k)
        if random.random() < c_rate:
            child1, child2 = perform_crossover(p1, p2, cmethod=cmethod) 
            children_list.extend([child1, child2])
        else:
            children_list.extend([p1.copy(), p2.copy()])
    
    children_list = children_list[:POPSIZE]
    
    #2. Đột biến trên con (mutation)
    for i in range(POPSIZE):
        if random.random() < m_rate:
            children_list[i] = apply_mutation(children_list[i], mmethod) 
        # --- THÊM LOCAL SEARCH TẠI ĐÂY ---
        # Chỉ áp dụng cho một số cá thể may mắn để tiết kiệm thời gian
        # if random.random() < ls_rate:
        #     # Lưu ý: local_search_softTW trả về cá thể đã tối ưu và tự tính lại fitness/objective
        #     # children_list[i] = local_search_softTW(children_list[i], problem, max_no_improve=5, imp_type=imp_type)
        #     # children_list[i] = local_search_softTW(children_list[i], problem, imp_type=imp_type)
        #     # local_search_softTW_best_improvement
        #     children_list[i] = local_search_softTW_first_improvement(children_list[i], problem)
            # children_list[i] = local_search_softTW_best_improvement(children_list[i], problem)
    
    #3. Gộp hai quần thể
    combined = children_list + [ind.copy() for ind in pop]
    
    #4. Tính toán fitness cho từng cá thể
    ranking_parameter = kwargs.get('ranking_f_parameter')
    calculate_fitness(combined, problem, fmethod, ranking_parameter=ranking_parameter)
    
    #5. Chọn lọc sinh tồn (survivors selection)
    # combined.sort(key=lambda x: x.fitness)
    best = min(combined, key=lambda x: x.fitness)
    trunc = kwargs.get('trunc_sv_parameter')
    pressure = kwargs.get('linear_sv_parameter')
    tourn_size = kwargs.get('tourn_sv_parameter') 
    new_pop = apply_sv_selection(combined, POPSIZE, svmethod,
                                 trunc_sv_parameter=trunc,
                                 linear_parameter=pressure,
                                 tourn_sv_parameter=tourn_size)
    # new_pop.insert(0,best.copy())
    new_pop.append(best.copy()) # giữ một cá thể tốt nhất từ thế hệ trước
    calculate_fitness(new_pop, problem, fmethod, ranking_parameter=ranking_parameter)
    new_pop.sort(key=lambda x: x.fitness)
    new_pop = new_pop[:POPSIZE]
    # calculate_fitness(new_pop, problem, fmethod, ranking_parameter=ranking_parameter)
    
    return new_pop
    
    

def run_genetic_algorithm(
                        problem, pop_size, 
                      c_rate, m_rate, 
                      generations, maximum_loop,
                    #   gen_type='greedy',
                    #   greedy_rate=0.2, search_size=2,
                      **kwargs):
    # params ={
    #     "fmethod": kwargs.get('fmethod'),
    #     "smethod": kwargs.get('smethod'),
    #     "cmethod": kwargs.get('cmethod'),
    #     "mmethod": kwargs.get('mmethod'),
    #     "svmethod": kwargs.get('svmethod'),
    #     "ranking_f_parameter": kwargs.get('ranking_f_parameter'),
    #     "tourn_s_parameter": kwargs.get('tourn_s_parameter'),
    #     "ranking_s_parameter": kwargs.get('rankins_s_parameter'),
    #     "trunc_sv_parameter": kwargs.get('trunc_sv_parameter'),
    #     "linear_sv_parameter": kwargs.get('linear_sv_parameter'),
    #     "tourn_sv_parameter": kwargs.get('tourn_sv_parameter') 
    # }
    params ={
        "fmethod": kwargs.get('fmethod', 'std'),
        "smethod": kwargs.get('smethod', 'tournament'),
        "cmethod": kwargs.get('cmethod', 'ox'),
        "mmethod": kwargs.get('mmethod', 'inversion'),
        "svmethod": kwargs.get('svmethod', 'truncation'),
        "ranking_f_parameter": kwargs.get('ranking_f_parameter', 0.5),
        "tourn_s_parameter": kwargs.get('tourn_s_parameter', 4),
        "ranking_s_parameter": kwargs.get('ranking_s_parameter', 10),
        "trunc_sv_parameter": kwargs.get('trunc_sv_parameter', 0.5),
        "linear_sv_parameter": kwargs.get('linear_sv_parameter', 1.5),
        "tourn_sv_parameter": kwargs.get('tourn_sv_parameter', 4),
        "ls_rate": kwargs.get('ls_rate', 0.1)
    }
    
    gen_type = kwargs.get('gen_type')
    greedy_rate = kwargs.get('greedy_rate', 0.5)
    search_size = kwargs.get('search_size', 2)
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
    
    # Tính fitness cho quần thể ban đầu    
    calculate_fitness(pop, problem, params["fmethod"], **{"ranking_f_parameter": params["ranking_f_parameter"]})
    
    last_pop_best_indi = min(pop, key=lambda x: x.fitness)
    last_fitness = last_pop_best_indi.fitness
    print ("Initial fitness: ", str(last_fitness))
    
    progress = []
    progress.append(last_fitness)
    
    current_pop = pop
    loop_not_improve = 0
    
    # Lưu lại m_rate để bắt đầu điều chỉnh
    initial_m_rate = m_rate
    
    for i in range(generations):
        
        # -----------------------------
        # Cài đặt params theo thế hệ (Adaptive parameters)
        # -----------------------------
        progress_pct = i / generations # Tiến độ (0.0->1.0)
        # # 1. Giảm dần mutaion rate (tuyến tính)
        # # Từ m_rate ban đầu giảm về m_rate/4 ở cuối
        # current_m_rate = initial_m_rate *(1 - 0.75 * progress_pct)
        
        # # 2. Thay đổi phương pháp Mutation và Selection theo giai đoạn
        # if progress_pct < 0.3:
        #     # Giai đoạn khám phá (Exploration)
        #     params['mmethod'] = 'scramble' # xáo trộn
        #     params['smethod'] = 'roulette' # Chọn theo xác suất (điều này giữ độ đa dạng)
        #     params['tourn_s_']
        
        # -----1. Điều chỉnh Parent Selection-----
        if progress_pct < 0.3:
            # Giai đoạn đầu cần đa dạng cao
            # Dùng Roulette: để cho phép cá yếu cũng có cơ hội được làm parent
            params['smethod'] = 'roulette'
        else:
            # Giai đoạn sau: Cần chọn lọc khắt khe hơn
            # Dùng Tournament: tranh đấu gây áp lực chọn lọc
            params['smethod'] = 'tournament'
            # gen đầu áp lực 3, gen cuối áp lực 8
            params['tourn_s_parameter'] = 3 + int(progress_pct * 2)
        # -----2. Điều chỉnh Survivor Selection-----
        if progress_pct < 0.4:                 
            # Giai đoạn đầu: Giữ lại độ đa dạng 
            # Dùng SUS (Stochastic Universal Sampling - tương tự như roulette)
            params['svmethod'] = 'sus'
        elif progress_pct < 0.8:
            # Giai đoạn giữa dùng Tournament:: tranh đấu gây áp lực chọn lọc
            params['svmethod'] = 'tournament'
            params['tourn_sv_parameter'] = 3 + int(progress_pct * 2)
        elif progress_pct < 0.95:
            # Giai đoạn giữa dùng Tournament:: tranh đấu gây áp lực chọn lọc
            # params['svmethod'] = 'tournament'
            params['tourn_sv_parameter'] = 2
        else:
            params['tourn_sv_parameter'] = int(pop_size*0.05)
        # else:
        #     # Giai đoạn sau: Loại bỏ mạnh những cá thể yếu
        #     # Dùng Truncation có giảm dần 
        #     params['svmethod'] = 'truncation'
        #     params['trunc_sv_parameter'] = 1 - (0.5 * progress_pct)
            
        # -----3. Điều chỉnh Mutation-----
        # Giảm mutation rate tuyến tính
        # current_m_rate = initial_m_rate *(1 - 0.75 * progress_pct)
        # current_m_rate = initial_m_rate *(1 - 0.75 * progress_pct)
        # current_m_rate = initial_m_rate * (1 - 0.7 * progress_pct)
        if progress_pct < 0.4:
            current_m_rate = initial_m_rate * (1 - 0.7 * progress_pct)
            params['mmethod'] = 'inversion' # Mạnh nhất (đảo ngược đoạn)
        elif progress_pct < 0.8:
            current_m_rate = initial_m_rate * (1 - 0.7 * progress_pct)
            params['mmethod'] = 'scramble' # Trung bình (xáo trộn trong đoạn)
        elif progress_pct < 0.95:
            # current_m_rate = 0.8
            current_m_rate = 0.5
            params['mmethod'] = 'swap'  # Nhẹ nhất (đổi chỗ hai request)
        else:
            current_m_rate = 0.1
        
        # ----------------------------------------
        # Tạo thế hệ mới với params đã cập nhật
        # ----------------------------------------
        # next_pop = create_next_population(current_pop, problem, c_rate, m_rate, **params)
        next_pop = create_next_population(current_pop, problem, c_rate, current_m_rate, **params)
        # current_fitness = min(next_pop, key=lambda x: x.fitness)
        curr_pop_best_indi = min(next_pop, key=lambda x: x.fitness)
        current_fitness = curr_pop_best_indi.fitness
        # if i%30:
        if i % 20 == 0:
            # print("Current fitness of generation ", i, ": ", str(current_fitness))
            print(f"Gen {i}: Fit={current_fitness:.0f} | Mut={params['mmethod']} | Sel={params['smethod']} | Surv={params['svmethod']}")
            
        progress.append(current_fitness)
        
        if abs(last_fitness - current_fitness) < 1e-9:
            loop_not_improve += 1
        else:
            last_fitness = current_fitness
            loop_not_improve = 0
        
        if loop_not_improve == maximum_loop:
            print(f"Stop at gen {i} due to stagnation")
            break
        
        current_pop  = next_pop
    
    # # Cá thể tốt nhất trong quần thể cuối cùng:
    # final_fittest_individual = min(current_pop, key=lambda x: x.fitness)
    # print("---> Fittest individual: ", final_fittest_individual.fitness)
    # print("-----Corresponding to the best individual-----")
    # print("---> Route:",  final_fittest_individual.route)
    # print("---> Total service time: ", final_fittest_individual.total_service_time)
    # # print("Number of time window violations: ", final_fittest_individual.valid.count(False))
    # print("Number of time window violations: ", sum(1 for x in final_fittest_individual.late if x != 0))
    # print("Total time of arrivals late: ", sum(final_fittest_individual.late))
    # print("Number of arrivals ealier than opening time: ", sum(1 for x in final_fittest_individual.wait if x != 0))
    # print("Total time of arrivals ealier than opening time:", sum(final_fittest_individual.wait))
    # Cá thể tốt nhất trong quần thể cuối cùng:
    final_fittest_individual = min(current_pop, key=lambda x: x.fitness)
    print("Running Final Local Search...")
    # Tăng max_no_improve lên để vét cạn kỹ hơn ở bước cuối cùng
    # final_fittest_individual = local_search_softTW(final_fittest_individual, problem, max_no_improve=20)
    print("---> Fittest individual: ", final_fittest_individual.fitness)
    print("-----Corresponding to the best individual-----")
    print("---> Route:",  final_fittest_individual.route)
    print("---> Total service time: ", final_fittest_individual.route_computing[2])
    # print("Number of time window violations: ", final_fittest_individual.valid.count(False))
    print("Number of time window violations: ", sum(1 for x in final_fittest_individual.route_computing[3] if x != 0))
    print("Total time of late arrivals: ", final_fittest_individual.route_computing[4])
    print("Number of arrivals ealier than opening time: ", sum(1 for x in final_fittest_individual.route_computing[5] if x != 0))
    print("Total time of arrivals ealier than opening time:", final_fittest_individual.route_computing[6])
    
    plt.plot(progress)
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.show()
    
    return current_pop
    
    
def run_nsga2(problem, pop_size, c_rate, m_rate, generations, maximum_loop, **kwargs):
    print("Starting NSGA-II for Multi-Objectiv Optimization...")
    
    gen_type = kwargs.get('gen_type', 'greedy')
    greedy_rate = kwargs.get('greedy_rate', 0.5)
    search_size = kwargs.get('search_size', 2)
    
    cmethod = kwargs.get('cmethod', 'ox')
    mmethod = kwargs.get('mmethod', 'inversion')
    params ={
        "cmethod": kwargs.get('cmethod', 'ox'),
        "mmethod": kwargs.get('mmethod', 'inversion'),
        "tourn_s_parameter": kwargs.get('tourn_s_parameter', 4),
    }
    
    if gen_type == 'random':
        pop = gen_pop_fully_random(problem, pop_size)
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
    
    calculate_mo_fitness(pop, problem)
    
    # sắp xếp các cá thể vào từng front
    current_fronts = fast_non_dominated_sorting(pop)
    for front in current_fronts:
        # tính khoảng cách quy tụ cho từng cá thể trong từng front
        crowding_distance_assignment(front)
    
    last_pop_best_indi = current_fronts[0][0]
    # last_f1 = last_pop_best_indi.f1
    # last_f2 = last_pop_best_indi.f2
    last_f1 = last_pop_best_indi.fitness[0]
    last_f2 = last_pop_best_indi.fitness[1]
    print (f"Initial fitness:  [service_time: {last_f1}, violations: {last_f2[0]}, total time of late arrivals: {last_f2[1]}")

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
        offspring[:pop_size]
        calculate_mo_fitness(offspring, problem)
        # gộp hai quần thể
        combined_pop = offspring + [ind.copy() for ind in current_pop]
        
        calculate_mo_fitness(combined_pop, problem)
        # Chọn lọc các cá thể tốt nhất cho quần thể mới
        current_pop = nsga2_sv_selection(combined_pop, pop_size)
        
        current_fronts = fast_non_dominated_sorting(current_pop)
        # crowding_distance_assignment(current_pop)
        
        # pareto_front = fronts[0]
        if i % 20 == 0 or i == generations - 1:
            pareto_front = current_fronts[0]
            # minimum_travel_time = min(pareto_front, key=lambda x: x.f1)
            minimum_travel_time = min(pareto_front, key=lambda x: x.fitness[0]).fitness[0]
            # minimum_number_of_late_arrivals =   min(pareto_front, key=lambda x: x.f2[0])
            # minimum_total_late_arrival_time = min(pareto_front, key=lambda x: x.f2[1])
            minimum_number_of_late_arrivals =   min(pareto_front, key=lambda x: x.fitness[1][0]).fitness[1][0]
            minimum_total_late_arrival_time = min(pareto_front, key=lambda x: x.fitness[1][1]).fitness[1][1]
            print(f'Gen: {i} | Pareto size: {len(pareto_front)}| Minimum travel time: {minimum_travel_time}| Minimum number of late arrivals: {minimum_number_of_late_arrivals}| Minimum total late arrival time: {minimum_total_late_arrival_time }')
        
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
    f2_list = [ind.fitness[1][0] for ind in final_pareto]
    
    plt.figure(figsize=(10,6))
    plt.scatter(f1_list, f2_list, c='red', label='Pareto Front')
    plt.xlabel('Objective 1: Total service time')
    plt.ylabel('Objective 2: Number of arrival time')
    plt.title('Pareto Front obtained by NSGA-II')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return current_pop



def run_gp_algorithm(problem, pop_size, c_rate, m_rate, generations, maximum_loop, **kwargs):
    print("Starting Genetic Programming...")
    
    # 1. Khởi tạo quần thể GP
    # Lưu ý: create_population cần được import từ gp_op
    current_pop = create_population(problem, pop_size, max_depth=6)
    
    # # 2. Đánh giá fitness ban đầu
    # for ind in current_pop:
    #     ind.calObjective(current_pop) # Hàm này giờ đã tự gọi simulate_tsptw
    calculate_fitness(current_pop, problem)    
    best_ind = min(current_pop, key=lambda x: x.fitness)
    print(f"Initial GP fitness: {best_ind.fitness}")
    
    loop_no_improve = 0
    last_fitness = best_ind.fitness
    progress = [last_fitness]
    
    for gen in range(generations):
        next_pop = []
        
        # Elitism: Giữ lại 1 cá thể tốt nhất
        next_pop.append(best_ind.copy())
        
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
            
            # Reset route cũ để cây mới sinh ra route mới
            child1.route = []
            child2.route = []
            
            next_pop.extend([child1, child2])
            
        # # Cắt tỉa về đúng pop_size
        # pop = next_pop[:pop_size]
        combined_pop = next_pop + [ind.copy() for ind in current_pop]
        # # Đánh giá quần thể mới
        # for ind in pop:
        #     ind.calObjective(problem)
        calculate_fitness(combined_pop, problem)
        
        combined_pop.sort(key=lambda x: x.fitness)
        
        current_pop = combined_pop[:pop_size]
        
        current_best = min(current_pop, key=lambda x: x.fitness)
        # In log định kỳ
        if gen % 10 == 0:
            print(f"Gen {gen}: Best Fitness = {current_best.fitness}")
            
        progress.append(current_best.fitness)
        
        # Kiểm tra điều kiện dừng
        if current_best.fitness < best_ind.fitness:
            best_ind = current_best.copy()
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
    
    print("Best GP Route:", best_ind.route)
    print("Best GP Rule:", best_ind.tree.to_string())
    
    return current_pop