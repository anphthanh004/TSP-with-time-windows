from collections import defaultdict
import random

def dominate(ind1, ind2):
    """
    Trả về True nếu ind1 dominate ind2.
    Mục tiêu: Minimize F1 (thời gian di chuyển), Minimize F2 (Số lần đến trễ)
    """
    # f1_a, f2_a = ind1.f1, ind1.f2
    # f1_b, f2_b = ind2.f1, ind2.f2
    f1_a, f2_a = ind1.fitness
    f1_b, f2_b = ind2.fitness
    
    # Dominate nếu tốt hơn hoặc bằng ở mọi mục tiêu và tốt hơn ít nhất 1 mục tiêu
    if (f1_a <= f1_b and f2_a <= f2_b) and (f1_a < f1_b or f2_a < f2_b):
        return True
    return False

def fast_non_dominated_sorting(population):
    fronts = [[]]
    # số lượng individual trội hơn một individual cụ thể
    domination_count = defaultdict(int)
    # tập hợp các individual bị trội bởi một individual cụ thể
    dominated_solutions = defaultdict(list)
    
    for p in population:
        for q in population:
            if p == q: continue
            if dominate(p, q):
                dominated_solutions[p].append(q)
            elif dominate(q, p):
                domination_count[p] += 1
        
        if domination_count[p] == 0:
            p.rank = 0 
            fronts[0].append(p)
            
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
        else:
            break
    
    # # xóa list rỗng cuối cùng nếu có:
    # if not fronts[-1]:
    #     fronts.pop()
            
    return fronts

def crowding_distance_assignment(front):
    l = len(front)
    if l == 0: return
    
    for ind in front:
        ind.distance = 0
        
    # Tính cho từng mục tiêu (2 mục tiêu)
    for m in range(2):
        if m == 0:
            front.sort(key=lambda x: x.fitness[0])
            f_min = front[0].fitness[0]
            f_max = front[-1].fitness[0]
        else:
            front.sort(key=lambda x: x.fitness[1][0])
            f_min = front[0].fitness[1][0]
            f_max = front[-1].fitness[1][0]
        # gán vô cùng cho 2 biên để luôn được giữ lại
        front[0].distance = float('inf')
        front[-1].distance = float('inf')
        
        # f_min = front[0].fitness[m]
        # f_max = front[-1].fitness[m]
        
        if f_max == f_min: continue
        norm = f_max - f_min
        if m == 0:
            for i in range(1, l-1):
                front[i].distance += (front[i+1].fitness[0] - front[i-1].fitness[0]) / norm
        else:
            for i in range(1, l-1):
                front[i].distance += (front[i+1].fitness[1][0] - front[i-1].fitness[1][0]) / norm            

# def nsga2_sv_selection(combined_pop, pop_size):
#     """Chọn k cá thể tốt nhất cho thế hệ sau"""
#     fronts = fast_non_dominated_sorting(combined_pop)
#     new_pop = []
    
#     for front in fronts:
#         crowding_distance_assignment(front)
#         front.sort(key=lambda x: x.distance, reverse=True)
        
#         if len(new_pop) + len(front) <= pop_size:
#             new_pop.extend(front)
#         else:
#             needed = pop_size - len(new_pop)
#             new_pop.extend(front[:needed])
#             break
            
#     return new_pop
def nsga2_sv_selection(combined_pop, pop_size):
    """
    Chọn lọc sinh tồn:
    1. Loại bỏ các cá thể trùng lặp
    2. Xếp hạng Front
    3. Lấy theo Rank và Crowding Distance
    """
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
    
    # seen_fitness = set()
    # for ind in combined_pop:
    #     if ind.fitness not in seen_fitness:
    #         seen_fitness.add(ind.fitness)
    #         unique_pop.append(ind)
    
    # if len(unique_pop) < pop_size:
    #     # bổ sung từ combined_pop
    #     for ind in combined_pop:
    #         if len(unique_pop) >= pop_size:
    #             break
    #         if ind not in unique_pop:
    #             unique_pop.append(ind)
    #     remaining = pop_size - len(unique_pop)
    #     unique_pop.extend(random.sample(combined_pop,remaining))
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
        

def nsga2_tourn_selection(pop, tour_size):
    """
    Chọn cha mẹ: So sánh 2 cá thể ngẫu nhiên.
    1. Rank nhỏ hơn (tốt hơn) được chọn
    2. Nếu rank bằng nhau, crowing distance lớn hơn (tức đa dạng hơn) được chọn
    """
    i1, i2 = random.sample(pop, 2)
    if i1.rank < i2.rank:
        return i1
    elif i2.rank < i1.rank:
        return i2
    else:
        if i1.distance > i2.distance:
            return i1
        else:
            return i2