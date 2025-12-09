import random
import numpy as np
from .nsga2_algorithm import crowding_distance_assignment, fast_non_dominated_sorting

def cal_moo_fitness(pop):
    for ind in pop:
        ind.calObjective()
        f1 = ind.objective[0]
        f2 = ind.objective[1]
        ind.fitness = (f1, f2)
        

# ----------------------------
# Crossover operators
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
# Mutation operators
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

# -------------------------
# Survivors selection
# ------------------------
def nsga2_sv_selection(combined_pop, pop_size):
    """
    Chọn lọc sinh tồn:
    1. Loại bỏ các cá thể trùng lặp
    2. Xếp hạng Front
    3. Lấy theo Rank và Crowding Distance
    """
    # """
    # Chọn lọc sinh tồn:
    # 1. Xếp hạng Front
    # 2. Lấy theo Rank và Crowding Distance
    # """
    # 1. Loại bỏ trùng lặp theo route
    unique = {}
    for ind in combined_pop:
        key = tuple(ind.route)
        if key not in unique:
            unique[key] = ind
    unique_pop = list(unique.values())
    if len(unique_pop) < pop_size:
        need = pop_size - len(unique_pop)
        extra = random.sample(combined_pop, need)
        unique_pop.extend(extra)
    
    # 2. Xếp hạng 
    fronts = fast_non_dominated_sorting(unique_pop)
    new_pop = []
    
    # 3. Chọn theo Front và Distance
    for front in fronts:
        crowding_distance_assignment(front)
        front.sort(key=lambda x: x.distance, reverse = True)
        
        if len(new_pop) + len(front) <= pop_size:
            new_pop.extend(front)
        else:
            needed = pop_size - len(new_pop)
            new_pop.extend(front[:needed])
            break
    
    return new_pop
        
# -------------------------
# Parents selection
# ------------------------
def nsga2_tourn_selection(pop, tour_size):
    """
    Chọn cha mẹ: So sánh 2 cá thể ngẫu nhiên.
    1. Rank nhỏ hơn (tốt hơn) được chọn
    2. Nếu rank bằng nhau, crowing distance lớn hơn (tức đa dạng hơn) được chọn
    """
    tourn = random.sample(pop, tour_size)
    # i1, i2 = random.sample(pop, 2)
    #     if i1.rank < i2.rank:
    #         return i1
    #     elif i2.rank < i1.rank:
    #         return i2
    #     else:
    #         if i1.distance > i2.distance:
    #             return i1
    #         else:
    #             return i2
    while len(tourn) > 1:
        i1, i2 = random.sample(tourn, 2)
        if i1.rank < i2.rank:
            tourn.remove(i2)
        elif i2.rank < i1.rank:
            tourn.remove(i1)
        else:
            if i1.distance > i2.distance:
                tourn.remove(i2)
            else:
                tourn.remove(i1)
    return tourn[0]