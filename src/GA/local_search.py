import numpy as np
import random
import time
from .ga_structures import Individual

# Hàm tính lại chi phí cho phần đuôi lộ trình
def recompute_suffix(route, problem, start_idx, prev_node, current_time):

    n = len(route)

    suffix_travel_sum = 0.0
    suffix_total_lateness = 0.0
    suffix_total_wait = 0.0

    prev = prev_node
    t = current_time

    tm = problem.time_matrix 
    for idx in range(start_idx, n):
        node = route[idx]
        e_i, l_i, d_i = problem.request[node - 1]

        travel = tm[prev][node]
        suffix_travel_sum += travel
        arrival = t + travel

        w = max(0.0, e_i - arrival)
        if arrival < e_i:
            arrival = e_i  
        suffix_total_wait += w
        
        late = max(0.0, arrival - l_i)
        suffix_total_lateness += late

        dep = arrival + d_i

        t = dep
        prev = node

    # Quay về depot
    return_travel = tm[prev][0]
    suffix_travel_sum += return_travel
    t += return_travel

    return suffix_travel_sum, suffix_total_lateness, suffix_total_wait


def calculate_total_cost(travel_sum, total_lateness, total_wait, problem):
    p_wait, p_late = problem.penalty
    return travel_sum + p_wait * total_wait + p_late * total_lateness


# ------------------------------------------------
# Local Search First Improvement
# ------------------------------------------------
def local_search_softTW_first_improvement(ind, max_iter=None, rotate_moves=True):
    problem = ind.problem
    best = ind.copy()
    best.calObjective()
    current_best_cost = best.objective

    n = problem.num_request

    # Tham số giới hạn neighborhood 
    K_default = max(10, min(20, n // 5))  # số lân cận mặc định
    RANDOM_SAMPLES = max(5, min(20, n // 10))  # số vị trí ngẫu nhiên thêm vào

    progress = []
    eval_count = 0
    start_time = time.perf_counter()

    move_order = ['relocate', 'swap', '2-opt']
    improved = True
    iteration = 0
    
    while improved:
        improved = False
        if max_iter and iteration >= max_iter:
            break
        iteration += 1
        
        res = best.compute_route_forward()
        current_travels = res[0]     
        current_departures = res[2]  
        current_lateness = res[4]    
        current_wait = res[6]       

        # Prefix sums để tránh tính lại nhiều lần
        prefix_travel = [0.0] * (n + 1)
        prefix_lat = [0.0] * (n + 1)
        prefix_wait = [0.0] * (n + 1)
        for k in range(n):
            prefix_travel[k + 1] = prefix_travel[k] + current_travels[k]
            prefix_lat[k + 1] = prefix_lat[k] + current_lateness[k]
            prefix_wait[k + 1] = prefix_wait[k] + current_wait[k]

        # Hàm lấy thông tin prefix (phần đầu giữ nguyên)
        def get_prefix_info(start_idx_):
            if start_idx_ == 0:
                return 0, 0.0, 0.0, 0.0, 0.0
            prev_node_ = best.route[start_idx_ - 1]
            dep_time_ = current_departures[start_idx_ - 1]
            pre_travel_ = prefix_travel[start_idx_]
            pre_lat_ = prefix_lat[start_idx_]
            pre_wait_ = prefix_wait[start_idx_]
            return prev_node_, dep_time_, pre_travel_, pre_lat_, pre_wait_

        route_len = n
        route_list = best.route 

        # Prepare shuffled indices for i to find improvement early
        indices_i = list(range(route_len))
        random.shuffle(indices_i)

        if rotate_moves:
            offset = (iteration - 1) % len(move_order)
            this_order = move_order[offset:] + move_order[:offset]
        else:
            this_order = move_order
            
        for op in this_order:
            if op == 'relocate':
                # 1. RELOCATE (Insert node i into position j)
                K = min(K_default, route_len - 1)
                for i in indices_i:
                    # chọn vị trí j để chèn node[i] trong lân cận K của i
                    start_j = max(0, i - K) # bán kính K
                    end_j = min(route_len, i + K + 1) # bán kính K
                    j_candidates = list(range(start_j, end_j))
                    # (i-1).i.(i+1) ta thấy là dù chèn i vào ngay trước i hay ngay sau i đều vô nghĩa 
                    j_candidates = [j for j in j_candidates if not (j == i or j == i + 1)]
                    # thêm 1 vài vị trí ngẫu nhiên để giữ độ đa dạng 
                    # (tạo list chứa các phần tử không trong j và có nghĩa -> lấy ngẫu nhiên từ list này với số lượng là min(RANDOM_SAMPLES, list đó) -> chèn các phần tử này vào j_candidates)
                    if route_len - len(j_candidates) - 1 > 0:
                        pool = [x for x in range(route_len) if x not in j_candidates and x != i and x != i + 1]
                        sample_count = min(RANDOM_SAMPLES, len(pool)) 
                        if sample_count > 0:
                            j_candidates.extend(random.sample(pool, sample_count))
                    random.shuffle(j_candidates)

                    for j in j_candidates:
                        rt = route_list[:] 
                        node = rt.pop(i)
                        rt.insert(j, node)
                        start_idx = min(i, j)
                        prev_n, dep_t, pre_travel, pre_lat, pre_wait = get_prefix_info(start_idx)
                        suf_travel, suf_lat, suf_wait = recompute_suffix(rt, problem, start_idx, prev_n, dep_t)
                        eval_count += 1
                        new_cost = calculate_total_cost(pre_travel + suf_travel,
                                                        pre_lat + suf_lat,
                                                        pre_wait + suf_wait, problem)
                        if new_cost < current_best_cost - 1e-9:
                            # log the move
                            progress.append({
                                'time': time.perf_counter() - start_time,
                                'iteration': iteration,
                                'move': 'relocate',
                                'i': i,
                                'j': j,
                                'old_cost': current_best_cost,
                                'new_cost': new_cost,
                                'improvement': current_best_cost - new_cost
                            })
                            best.route = rt
                            best.fitness = new_cost
                            best.objective = new_cost
                            current_best_cost = new_cost
                            improved = True
                            break  # First Improvement: stop early
                    if improved:
                        break
                if improved:
                    continue
            elif op == 'swap':                
                # 2. SWAP (swap i and j)
                # lặp ngẫu nhiên các chỉ số
                indices_i = list(range(route_len))
                random.shuffle(indices_i)
                for i in indices_i:
                    j_candidates = list(range(i + 1, min(route_len, i + 1 + K)))
                    pool = [x for x in range(i + 1, route_len) if x not in j_candidates]
                    sample_count = min(RANDOM_SAMPLES, len(pool)) # sóo lượng vị 
                    if sample_count > 0:
                        j_candidates.extend(random.sample(pool, sample_count))
                    random.shuffle(j_candidates)

                    for j in j_candidates:
                        rt = route_list[:]
                        rt[i], rt[j] = rt[j], rt[i]
                        start_idx = min(i, j)
                        prev_n, dep_t, pre_travel, pre_lat, pre_wait = get_prefix_info(start_idx)
                        suf_travel, suf_lat, suf_wait = recompute_suffix(rt, problem, start_idx, prev_n, dep_t)
                        eval_count += 1
                        new_cost = calculate_total_cost(pre_travel + suf_travel,
                                                        pre_lat + suf_lat,
                                                        pre_wait + suf_wait, problem)
                        if new_cost < current_best_cost - 1e-9:
                            progress.append({
                                'time': time.perf_counter() - start_time,
                                'iteration': iteration,
                                'move': 'swap',
                                'i': i,
                                'j': j,
                                'old_cost': current_best_cost,
                                'new_cost': new_cost,
                                'improvement': current_best_cost - new_cost
                            })
                            best.route = rt
                            best.fitness = new_cost
                            best.objective = new_cost
                            current_best_cost = new_cost
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    continue
            else: 
                # 3. 2-OPT (reverse segment i..j)
                indices_i = list(range(route_len - 1))
                random.shuffle(indices_i)
                for i in indices_i:
                    j_candidates = list(range(i + 1, min(route_len, i + 1 + K)))
                    pool = [x for x in range(i + 1, route_len) if x not in j_candidates]
                    sample_count = min(RANDOM_SAMPLES, len(pool))
                    if sample_count > 0:
                        j_candidates.extend(random.sample(pool, sample_count))
                    random.shuffle(j_candidates)

                    for j in j_candidates:
                        rt = route_list[:]
                        rt[i:j + 1] = rt[i:j + 1][::-1]
                        start_idx = i
                        prev_n, dep_t, pre_travel, pre_lat, pre_wait = get_prefix_info(start_idx)
                        suf_travel, suf_lat, suf_wait = recompute_suffix(rt, problem, start_idx, prev_n, dep_t)
                        eval_count += 1
                        new_cost = calculate_total_cost(pre_travel + suf_travel,
                                                        pre_lat + suf_lat,
                                                        pre_wait + suf_wait, problem)
                        if new_cost < current_best_cost - 1e-9:
                            progress.append({
                                'time': time.perf_counter() - start_time,
                                'iteration': iteration,
                                'move': '2-opt',
                                'i': i,
                                'j': j,
                                'old_cost': current_best_cost,
                                'new_cost': new_cost,
                                'improvement': current_best_cost - new_cost
                            })
                            best.route = rt
                            best.fitness = new_cost
                            best.objective = new_cost
                            current_best_cost = new_cost
                            improved = True
                            break
                    if improved:
                        break

    best.calObjective()
    return best, progress


