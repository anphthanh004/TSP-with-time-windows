from collections import defaultdict
import random

def dominate(ind1, ind2):
    """
    Trả về True nếu ind1 dominate ind2.
    Mục tiêu: Minimize F1 (thời gian di chuyển), Minimize F2 (thời gian đến sớm + đến trễ)
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
            
    return fronts

def crowding_distance_assignment(front):
    l = len(front)
    if l == 0: return
    
    for ind in front:
        ind.distance = 0
        
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
        
        if f_max == f_min: continue
        norm = f_max - f_min
        if m == 0:
            for i in range(1, l-1):
                front[i].distance += (front[i+1].fitness[0] - front[i-1].fitness[0]) / norm
        else:
            for i in range(1, l-1):
                front[i].distance += (front[i+1].fitness[1][0] - front[i-1].fitness[1][0]) / norm            
