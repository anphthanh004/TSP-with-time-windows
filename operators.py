import random
import numpy as np
#----------------------------------
# 1. Fitness evaluatation operators
# ---------------------------------
def calculate_fitness(pop, problem, fmethod='std', **kwargs):
    for ind in pop:
        ind.calObjective(problem)
    if fmethod is None: fmethod = 'std'
    if fmethod == 'std':
        return cal_std_fitness(pop)
    elif fmethod == 'ranking':
        # p = kwargs.get('ranking_parameter', 0.5)
        p = kwargs.get('ranking_f_parameter')
        if p is None:
            p = 0.5
        return cal_ranking_fitness(pop, p)
    elif fmethod == 'distribution':
        return cal_dis_fitness(pop)
    else:
        raise ValueError(f"Invalid fitness calculating method {fmethod}."
                         "Valid options: 'std', 'ranking', 'distribution'")

def calculate_mo_fitness(pop, problem):
    for ind in pop:
        ind.calObjective(problem)
        f1 = ind.objective[0]
        f2 = ind.objective[1], ind.objective[2]
        # ind.f1 = f1
        # ind.f2 = f2
        ind.fitness = (f1, f2)
    # else:
    #     raise ValueError(f"Invalid fitness calculating method {fmethod}."
    #                      "Valid options: 'std', 'ranking', 'distribution'")
"""
a, standard fitness
"""
# def cal_std_fitness(indi_list, problem_type="GA"):
def cal_std_fitness(pop):
    # obj_values = []
    # for indi in indi_list:
    #     obj = indi.calObjective()  
    for indi in pop:
        # if problem_type == "GA":
        indi.fitness = indi.objective
            
"""
b, ranking fitness
"""
# def cal_ranking_fitness(indi_list, problem_type="GA", p = 0.5):
def cal_ranking_fitness(pop, p=0.5):
    pop = sorted(pop, key=lambda x: x.objective, reverse = True)
    #   total = 0 # tổng độ thích nghi, để chuẩn hóa
    for i,indi in enumerate(pop):
        #   if problem_type == "GA":
        indi.fitness = p*((1-p)**i) # càng xuất hiện sớm thì fitness càng cao, mục tiêu: individual có fitness càng bé thì càng tốt 
                                    # về sau fitness bé dần, do đó phải sắp xếp objective lớn lên đầu
    #     total += indi.fitness

    #   for indi in indi_list:
    #       indi.fitness = indi.fitness / total

"""
c, distribution fitness (fitness base on distribution)
"""
def cal_dis_fitness(pop):
    # obj_values = [indi.calObjective() for indi in pop]
    obj_values = [indi.objective for indi in pop]
    # obj_sorted_idx = sorted(range(len(obj_values)), key=lambda x: obj_values[x], reverse = True)
    obj_sorted_idx = sorted(range(len(obj_values)), key=lambda x: obj_values[x])
    obj_ranks = [0] * len(pop)
    for rank, idx in enumerate(obj_sorted_idx):
        obj_ranks[idx] = rank + 1

    pl_values =[0]*len(pop) # pl càng bé thì cá thể phân ly càng mạnh -> rank cao (số nhỏ)
    for i in range(len(pop)):
        for j in range(len(pop)):
            if i != j:
                if obj_values[i] == obj_values[j]:
                    pl_values[i] += 1e6
                else:
                    pl_values[i] += 1 / abs(obj_values[i] - obj_values[j])
    
    pl_sorted_idx = sorted(range(len(pl_values)), key=lambda x: pl_values[x])
    pl_ranks = [0] * len(pop)
    for rank, idx in enumerate(pl_sorted_idx):
        pl_ranks[idx] = rank + 1
    for i, indi in enumerate(pop):
        # rank_sum = obj_ranks[obj_values[i]] + pl_ranks[pl_values[i]]
        rank_sum = obj_ranks[i] + pl_ranks[i]
        indi.fitness = rank_sum


#------------------------------
# 2.Parent seletion operators
#------------------------------
def select_parents(pop, smethod='tournament', **kwargs):
    if smethod is None: smethod = 'tournament'
    if smethod == 'random':
        return select_parents_random(pop)
    elif smethod == 'roulette':
        return select_parents_roulette(pop)
    elif smethod == 'tournament':
        # tourn_size = kwargs.get('tourn_s_parameter', 4)
        tourn_size = kwargs.get('tourn_s_parameter')
        if tourn_size is None:
            tourn_size = 4
        return select_parents_tour(pop, tourn_size)
    elif smethod == 'ranking':
        k = max(int(len(pop) * 0.3), 3)
        top_k = kwargs.get('ranking_s_parameter')
        if top_k is None:
            top_k = k
        return select_parents_ranking(pop, k=top_k)
    else:
        raise ValueError(f"Invalid parent selection method '{smethod}'. "
                         "Valid options: 'random', 'roulette', 'tournament', 'ranking'.")

"""
a, random selection
"""
def select_parents_random(pop):
  p1,p2 = random.sample(pop, 2)
  return p1,p2

"""
b, roulette selection (base on fitness) 
"""
def select_parents_roulette(pop):
    inv_fit = [1/indi.fitness for indi in pop]
    sum_inv_fit = sum(inv_fit)
    inv_fit_norm = [f/sum_inv_fit for f in inv_fit]
    cum_fit = np.cumsum(inv_fit_norm)
    
    r1, r2 = np.random.rand(2) # lấy 2 số ngẫu nhiên trong (0,1)
    id1 = np.searchsorted(cum_fit, r1)
    id2 = np.searchsorted(cum_fit, r2)
    p1, p2 = pop[id1], pop[id2]
    return p1, p2

"""
c, tournament selection (base on fitness)
"""
# áp dụng được cho cả tiêu chuẩn hay thứ hạng
def select_parents_tour(pop, k=4):
    tour = random.sample(pop,k)
    # Chọn cá thể có fitness tốt nhất trong mỗi nhóm
    mid = k//2
    tour1 = tour[:mid]
    tour2 = tour[mid:]
    p1 = min(tour1, key=lambda x: x.fitness)
    p2 = min(tour2, key=lambda x: x.fitness)
    return p1, p2

"""
d, ranking selection (base on fitness)
"""
def select_parents_ranking(pop, k = 4):
  pop = sorted(pop, key = lambda x: x.fitness)
  # lấy top k phần tử
  parent = pop[:k]
  p1, p2 = random.sample(parent, 2)
  return p1, p2


# ----------------------------
# 3. Crossover operators
# ----------------------------
def perform_crossover(parent1, parent2, cmethod='ox'):
    if cmethod is None: cmethod = "ox"
    if cmethod == 'ox':
        return perform_ox_crossover(parent1, parent2)
    elif cmethod == 'pmx':
        return perform_pmx_crossover(parent1, parent2)
    elif cmethod == 'cx':
        return perform_cx_crossover(parent1, parent2)
    else:
        raise ValueError(f"Invalid crossover method '{cmethod}'. "
                         "Valid options: 'ox', 'pmx', 'cx'.")
        
    

"""
a, Order crossover (OX) - Lai ghép thứ tự
"""

def perform_ox_crossover(parent1, parent2):
    n = parent1.problem.num_request
    p1 = parent1.route
    p2 = parent2.route
    c1 = [None] * n
    c2 = [None] * n
    start = np.random.randint(0,n -1)
    end = np.random.randint(start+1, n)

    c1[start:end+1] = p1[start:end+1]
    c2[start:end+1] = p2[start:end+1]

    # Điền phần còn lại từ các parent
    pos1 = pos2 = (end + 1) % n
    for i in range(n):
        gene2 = parent2.route[i]
        gene1 = parent1.route[i]
        if gene2 not in c1:
            c1[pos1] = gene2
            pos1 = (pos1 + 1) % n
        if gene1 not in c2:
            c2[pos2] = gene1
            pos2 = (pos2 + 1) % n
    child1 = parent1.copy()
    child2 = parent2.copy()
    child1.route = c1
    child2.route = c2
    return child1, child2

"""
b, Partially map crossover (PMX) - Lai ghép ánh xạ từng phần
"""
def perform_pmx_crossover(parent1, parent2):
    n = parent1.problem.num_request
    p1 = parent1.route
    p2 = parent2.route
    c1 = [None] * n
    c2 = [None] * n
    start = np.random.randint(0,n -1)
    end = np.random.randint(start+1, n)

    c1[start:end+1] = p1[start:end+1]
    c2[start:end+1] = p2[start:end+1]
    
    mapping1 = {p1[i]: p2[i] for i in range(start, end+1)}
    mapping2 = {p2[i]: p1[i] for i in range(start, end+1)}

    for i in range(n):
        if i >= start and i <= end:
            continue
        gene = p2[i]
        while gene in c1:
            gene = mapping1[gene]
        c1[i] = gene

    for i in range(n):
        if i >= start and i <= end:
            continue
        gene = p1[i]
        while gene in c2:
            gene = mapping2[gene]
        c2[i] = gene

    child1 = parent1.copy()
    child2 = parent2.copy()
    child1.route = c1
    child2.route = c2
    return child1, child2


"""
c, Cycle crossover (CX) - Lai ghép chu trình
"""
def perform_cx_crossover(parent1, parent2):
    n = parent1.problem.num_request
    p1 = parent1.route
    p2 = parent2.route
    c1 = [None] * n
    c2 = [None] * n

    # Tìm chu trình (đầu tiên và chỉ 1 chu trình này) bắt đầu từ vị trí 0
    cycle_positions = []
    index = 0  # bắt đầu từ vị trí 0 (tức gen 1)
    while index not in cycle_positions:
        cycle_positions.append(index)
        value_in_p2 = p2[index]          # lấy gen tương ứng ở p2
        index = p2.index(value_in_p2)    # tìm xem gen đó nằm ở vị trí nào trong p1

    # sau khi tìm được 1 chu trình thì gán các vị trí trong chu trình tương ứng vào con
    for i in cycle_positions:
        c1[i] = p1[i]
        c2[i] = p2[i]

    # Các vị trí còn lại thì đảo nguồn từ cha sang con
    for i in range(n):
        if c1[i] is None:
            c1[i] = p2[i]
            c2[i] = p1[i]

    child1 = parent1.copy()
    child2 = parent2.copy()
    child1.route = c1
    child2.route = c2
    return child1, child2

# ---------------------------
# 3. Mutation operators
# ---------------------------
def apply_mutation(child, mmethod='inversion'):
    if mmethod is None: mmethod = 'inversion'
    if mmethod == 'swap':
        return apply_swap_mutation(child)
    elif mmethod == 'scramble':
        return apply_scramble_mutation(child)
    elif mmethod == 'inversion':
        return apply_inversion_mutation(child)
    else:
        raise ValueError(f"Invalid mutation method {mmethod}."
                         "Valid options: 'swap', 'scramble', 'inversion'.")
"""
a, Swap mutation
"""
def apply_swap_mutation(child):
    n = child.problem.num_request
    pos1, pos2 = random.sample(range(n), 2) # vd: [5,1]  mỗi lần lấy không lặp lại 1 phần tử trong arr, trả về một arr mới 
    child.route[pos1], child.route[pos2] = child.route[pos2], child.route[pos1]
    return child

"""
b, Scramble mutation
"""
def apply_scramble_mutation(child):
    n = child.problem.num_request
    start = np.random.randint(0,n -1)
    end = np.random.randint(start+1, n)

    child.route[start:end+1] = random.sample(child.route[start:end+1], (end-start+1))
    return child

"""
c, Inversion mutation
"""
def apply_inversion_mutation(child):
    n = child.problem.num_request

    start = np.random.randint(0,n -1)
    end = np.random.randint(start+1, n)

    child.route[start:end+1] = child.route[start:end+1][::-1]
    return child


# -------------------------------------
# 4. Survivor selection operators
# -------------------------------------

def apply_sv_selection(combined, pop_size, svmethod='tournament', **kwargs):
    if svmethod is None: svmethod = 'tournament'
    if svmethod == 'truncation':
        # trunc = kwargs.get('truncation_parameter', 0.5)
        trunc = kwargs.get('trunc_sv_parameter')
        if trunc is None:
            trunc = 0.5
        return apply_trunc_sv_selection(combined, pop_size, trunc)
    elif svmethod == 'sus':
        return apply_sus_sv_selection(combined, pop_size)
    elif svmethod == 'linear':
        # pressure = kwargs.get('linear_parameter', 1.5)
        pressure = kwargs.get('linear_sv_parameter')
        if pressure is None:
            pressure = 1.5
        return apply_linear_sv_selection(combined, pop_size, pressure)
    elif svmethod == 'tournament':
        # tourn_size = kwargs.get('tourn_sv_parameter', 4) 
        tourn_size = kwargs.get('tourn_sv_parameter')
        if tourn_size is None:
            tourn_size = 4
        return apply_tour_sv_selection(combined, pop_size, tourn_size)
    else:
        raise ValueError(f"Invalid survivor selection method {svmethod}."
                         "Valid options: 'truncation', 'sus', 'linear', 'tournament'.")
        
    

"""
a, truncation survivor selection - chọn lọc cắt xén
"""
def apply_trunc_sv_selection(combined, pop_size, trunc=0.5):
    combined = sorted(combined, key = lambda x: x.fitness)
    # n = len(pop)
    draw = int(trunc*pop_size)
    new_pop = combined[:draw]
    while len(new_pop) < pop_size:
        indi = random.choice(new_pop)
        new_pop.append(indi.copy())
    return new_pop

"""
b, Stochastic universal sampling survivor selection(SUS) - Chọn lọc theo kiểu rải
"""
def apply_sus_sv_selection(combined, pop_size): 
    # n = len(pop)
    inv_fit = [1/indi.fitness for indi in combined]
    sum_inv_fit = sum(inv_fit)
    inv_fit_norm = [f/sum_inv_fit for f in inv_fit]

    cum_fit = np.cumsum(inv_fit_norm)
    new_pop = []
    r = np.random.rand()
    for _ in range(pop_size):
        ind = combined[np.searchsorted(cum_fit, r)]
        new_pop.append(ind.copy())
        if r + 1.0/pop_size < 1:
            r += 1.0/pop_size
        else:
            r += 1.0/pop_size - 1
    return new_pop

"""
c, Linear survivor selection (base on rank)
- Khả năng một cá thể được chọn chỉ dựa trên thứ hạng, không còn nhạy cảm với độ lớn giá trị thích nghi như bánh xe Roulette,...
- Ưu điểm:
  + Giảm thiểu rủi ro ưu tiên quá mức các cá thể tốt ở đầu quá trình tìm kiếm
  + Chỉ cần so sánh hơn kém tương đối giữa các cá thể -> giảm chi phí
"""
# chọn lọc tuyến tính
# chỉ áp dụng khi đánh giá độ thích nghi theo xếp hạng
def apply_linear_sv_selection(combined, pop_size, P=1.5): 
    # P là hệ số phóng đại [1.0,2.0] xác định áp lực lựa chọn
    # P = 1.0 -> mọi cá thể có xác suất bằng nhau
    # P = 2.0 -> cá thể tốt có xác suất gấp đôi trung bình
  
    # fit nhỏ -> tốt -> rank cao (1 là cao nhất)
    # new_fit tính dựa trên rank -> new_fit cũng nhỏ -> xác suất tích lũy nhỏ
    # -> cần nghịch đảo new_fit để cá thể tốt có xác suất cao được chọn theo roulette
    fit = [indi.fitness for indi in combined]
    fit_sorted_idx = sorted(range(pop_size), key=lambda x: fit[x])
    rank = [idx + 1 for idx in fit_sorted_idx]
    new_fit = [2-P+2*(P-1)*((x-1)/(pop_size-1)) for x in rank]
    inv_new_fit = [1/f for f in new_fit]
    sum_inv_new_fit = sum(inv_new_fit)
    inv_new_fit_norm = [f/sum_inv_new_fit for f in inv_new_fit]
    # phần dưới làm như chọn theo bánh xe Roulette
    cum_fit = np.cumsum(inv_new_fit_norm)
    new_pop = []
    for _ in range(pop_size):
        r = np.random.rand()
        ind = combined[np.searchsorted(cum_fit, r)]
        new_pop.append(ind.copy())
    return new_pop

"""
d, Tournament survivor selection
"""
def apply_tour_sv_selection(combined, pop_size, k = 4): 
#   n = len(pop)
  new_pop = []
  for _ in range(pop_size):
    tourn_size = min(len(combined), k) 
    tour = random.sample(combined,tourn_size)
    winner = min(tour, key=lambda ind: ind.fitness)
    new_pop.append(winner.copy())
  return new_pop