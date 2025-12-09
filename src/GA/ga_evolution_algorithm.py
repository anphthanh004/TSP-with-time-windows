import random
# import numpy
import matplotlib.pyplot as plt
from .ga_operators import cal_std_fitness, \
                        select_parents, apply_mutation, perform_crossover, apply_sv_selection
from .ga_initialization import gen_pop, gen_pop_fully_random, \
                            gen_pop_greedy1, gen_pop_greedy2,\
                            gen_pop_greedy3, gen_pop_greedy4



from .local_search import local_search_softTW_best_improvement, local_search_softTW_first_improvement
def create_next_population(pop, problem, c_rate, m_rate, **kwargs):
    POPSIZE = len(pop)
    children_list = []
    # lấy kiểu
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
    # ranking_parameter = kwargs.get('ranking_f_parameter')
    cal_std_fitness(combined)
    #5. Chọn lọc sinh tồn (survivors selection)
    best = min(combined, key=lambda x: x.fitness)
    trunc = kwargs.get('trunc_sv_parameter')
    tourn_size = kwargs.get('tourn_sv_parameter') 
    new_pop = apply_sv_selection(combined, POPSIZE, svmethod,
                                 trunc_sv_parameter=trunc,
                                 tourn_sv_parameter=tourn_size)
    new_pop.append(best.copy()) # giữ một cá thể tốt nhất từ thế hệ trước
    cal_std_fitness(new_pop)
    new_pop.sort(key=lambda x: x.fitness)
    new_pop = new_pop[:POPSIZE]
    
    return new_pop
    
    

def run_genetic_algorithm(
                        problem, pop_size, 
                      c_rate, m_rate, 
                      generations, maximum_loop,
                      **kwargs):
    params ={
        "smethod": kwargs.get('smethod', 'tournament'),
        "cmethod": kwargs.get('cmethod', 'ox'),
        "mmethod": kwargs.get('mmethod', 'inversion'),
        "svmethod": kwargs.get('svmethod', 'truncation'),
        "tourn_s_parameter": kwargs.get('tourn_s_parameter', 4),
        "ranking_s_parameter": kwargs.get('ranking_s_parameter', 10),
        "trunc_sv_parameter": kwargs.get('trunc_sv_parameter', 0.5),
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
    # calculate_fitness(pop, problem, params["fmethod"], **{"ranking_f_parameter": params["ranking_f_parameter"]})
    cal_std_fitness(pop)
    
    last_pop_best_indi = min(pop, key=lambda x: x.fitness)
    last_fitness = last_pop_best_indi.fitness
    print ("Initial fitness: ", str(last_fitness))
    
    progress = [sum(last_pop_best_indi.route_computing[0])]
    # progress.append(last_fitness)
    
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
        if i % 20 == 0:
            print(f"Gen {i}: Fit={current_fitness:.0f} | Mut={params['mmethod']} | Sel={params['smethod']} | Surv={params['svmethod']}")
        
        # progress = [sum(last_pop_best_indi.route_computing[0])]
        progress.append(sum(curr_pop_best_indi.route_computing[0]))
        
        if abs(last_fitness - current_fitness) < 1e-9:
            loop_not_improve += 1
        else:
            last_fitness = current_fitness
            loop_not_improve = 0
        
        if loop_not_improve == maximum_loop:
            print(f"Stop at gen {i} due to stagnation")
            break
        
        current_pop  = next_pop
    final_fittest_individual = min(current_pop, key=lambda x: x.fitness)
    # print("Running Final Local Search...")
    # Tăng max_no_improve lên để vét cạn kỹ hơn ở bước cuối cùng
    # final_fittest_individual = local_search_softTW(final_fittest_individual, problem, max_no_improve=20)
    print("---> Fittest individual: ", final_fittest_individual.fitness)
    print("-----Corresponding to the best individual-----")
    print("---> Route:",  final_fittest_individual.route)
    print("---> Total service time: ", final_fittest_individual.route_computing[3])
    print("---> Total travel time: ", sum(final_fittest_individual.route_computing[0]))
    # print("Number of time window violations: ", final_fittest_individual.valid.count(False))
    print("Number of time window violations: ", sum(1 for x in final_fittest_individual.route_computing[4] if x != 0))
    print("Total time of late arrivals: ", final_fittest_individual.route_computing[5])
    print("Number of arrivals ealier than opening time: ", sum(1 for x in final_fittest_individual.route_computing[6] if x != 0))
    print("Total time of arrivals ealier than opening time:", final_fittest_individual.route_computing[7])
    
    # plt.plot(progress)
    # plt.ylabel('Fitness')
    # plt.xlabel('Generation')
    # plt.show()
    
    return current_pop, progress