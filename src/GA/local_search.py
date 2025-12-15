from .ga_operators import cal_std_fitness
import random
from collections import deque

def recompute_suffix(route, problem, start_idx, prev_node, current_time):
    """
    Tính lại các thông số cho phần đuôi (suffix) của lộ trình từ start_idx.
    """
    n = problem.num_request
    arrivals = []
    departures = []
    lateness = []
    wait = []
    prev = prev_node
    t = current_time
    
    # Duyệt từ điểm thay đổi đến hết lộ trình
    for idx in range(start_idx, n):
        node = route[idx]
        e_i, l_i, d_i = problem.request[node-1]
        travel = problem.time_matrix[prev][node]
        
        arrival = t + travel
        
        # Tính toán wait và arrival thực tế
        w = max(0.0, e_i - arrival)
        if arrival < e_i:
            arrival = e_i
            
        # Tính toán late
        late = max(0.0, arrival - l_i)
        
        dep = arrival + d_i

        arrivals.append(arrival)
        lateness.append(late)
        wait.append(w)
        departures.append(dep)

        t = dep
        prev = node

    # Quay về depot
    t += problem.time_matrix[prev][0]
    suffix_total_time = t
    suffix_total_lateness = sum(lateness)
    suffix_total_wait = sum(wait)
    
    return arrivals, departures, suffix_total_time, lateness, suffix_total_lateness, wait, suffix_total_wait

def calculate_route_cost_from_metrics(total_time, total_lateness, total_wait, problem):
    p_wait, p_late = problem.penalty
    return total_time + p_wait * total_wait + p_late * total_lateness

# ---------------------------------------
# Main Local Search (Best-Improvement)
# ---------------------------------------
def local_search_softTW_best_improvement(ind, problem, max_no_improve=5):
    best = ind.copy()
    best.calObjective() 
    best_cost = best.objective

    n = problem.num_request
    no_improve = 0
    
    while no_improve < max_no_improve:
        improved = False
        
        # 1. Tính toán trạng thái hiện tại đầy đủ
        # Index 0-travels, 1-arrivals, 2-departures, 3-total_time, 4-lateness, 5-total_lateness, 6-wait, 7-total_wait
        _, _, departures, _, lateness, total_lateness, wait, total_wait = best.compute_route_forward(best.route, problem)
        
        prefix_departures = departures
        prefix_lateness = lateness
        prefix_wait = wait # Cần thêm danh sách wait

        best_move = None # (move_type, new_route, new_cost, ...)

        # Helper lấy thông tin prefix
        def get_prefix_info(start_idx_):
            if start_idx_ == 0:
                return 0, 0, 0.0, 0.0
            prev_node = best.route[start_idx_-1]
            prefix_time = prefix_departures[start_idx_-1]
            prefix_lat_sum = sum(prefix_lateness[:start_idx_])
            prefix_wait_sum = sum(prefix_wait[:start_idx_]) # Tính tổng wait phía trước
            return prev_node, prefix_time, prefix_lat_sum, prefix_wait_sum

        # --- Relocate ---
        for i in range(n):
            for j in range(n+1):
                if i == j or i == j-1: continue
                
                rt = best.route
                node = rt[i]
                temp_route = rt[:i] + rt[i+1:]
                
                if j <= i:
                    new_route = temp_route[:j] + [node] + temp_route[j:]
                    start_idx = j
                else:
                    new_route = temp_route[:j-1] + [node] + temp_route[j-1:]
                    start_idx = i

                prev_node, prefix_time, prefix_lat_sum, prefix_wait_sum = get_prefix_info(start_idx)
                
                _, _, suf_final_time, _, suf_total_lat, _, suf_total_wait = recompute_suffix(
                    new_route, problem, start_idx, prev_node, prefix_time
                )
                
                cand_total_lat = prefix_lat_sum + suf_total_lat
                cand_total_wait = prefix_wait_sum + suf_total_wait
                cand_cost = calculate_route_cost_from_metrics(suf_final_time, cand_total_lat, cand_total_wait, problem)

                if cand_cost < best_cost - 1e-9:
                    best_cost = cand_cost
                    best_move = (new_route, cand_cost)

        # --- Swap ---
        for i in range(n-1):
            for j in range(i+1, n):
                if best.route[i] == best.route[j]: continue
                
                new_route = best.route[:]
                new_route[i], new_route[j] = new_route[j], new_route[i]
                start_idx = i
                
                prev_node, prefix_time, prefix_lat_sum, prefix_wait_sum = get_prefix_info(start_idx)
                
                _, _, suf_final_time, _, suf_total_lat, _, suf_total_wait = recompute_suffix(
                    new_route, problem, start_idx, prev_node, prefix_time
                )
                
                cand_total_lat = prefix_lat_sum + suf_total_lat
                cand_total_wait = prefix_wait_sum + suf_total_wait
                cand_cost = calculate_route_cost_from_metrics(suf_final_time, cand_total_lat, cand_total_wait, problem)

                if cand_cost < best_cost - 1e-9:
                    best_cost = cand_cost
                    best_move = (new_route, cand_cost)

        # --- 2-opt ---
        for i in range(n-1):
            for j in range(i+1, n):
                rt = best.route
                new_route = rt[:i] + rt[i:j+1][::-1] + rt[j+1:]
                start_idx = i
                
                prev_node, prefix_time, prefix_lat_sum, prefix_wait_sum = get_prefix_info(start_idx)
                
                _, _, suf_final_time, _, suf_total_lat, _, suf_total_wait = recompute_suffix(
                    new_route, problem, start_idx, prev_node, prefix_time
                )
                
                cand_total_lat = prefix_lat_sum + suf_total_lat
                cand_total_wait = prefix_wait_sum + suf_total_wait
                cand_cost = calculate_route_cost_from_metrics(suf_final_time, cand_total_lat, cand_total_wait, problem)

                if cand_cost < best_cost - 1e-9:
                    best_cost = cand_cost
                    best_move = (new_route, cand_cost)

        # --- Or-Opt ---
        k = 2
        if n >= k:
            for i in range(n-k+1):
                for j in range(n+1):
                    if j >= i and j <= i+k: continue
                    
                    rt = best.route
                    block = rt[i:i+k]
                    temp_route = rt[:i] + rt[i+k:]
                    
                    if j <= i:
                        new_route = temp_route[:j] + block + temp_route[j:]
                        start_idx = j
                    else:
                        new_route = temp_route[:j-k] + block + temp_route[j-k:]
                        start_idx = i

                    prev_node, prefix_time, prefix_lat_sum, prefix_wait_sum = get_prefix_info(start_idx)
                    
                    _, _, suf_final_time, _, suf_total_lat, _, suf_total_wait = recompute_suffix(
                        new_route, problem, start_idx, prev_node, prefix_time
                    )
                    
                    cand_total_lat = prefix_lat_sum + suf_total_lat
                    cand_total_wait = prefix_wait_sum + suf_total_wait
                    cand_cost = calculate_route_cost_from_metrics(suf_final_time, cand_total_lat, cand_total_wait, problem)

                    if cand_cost < best_cost - 1e-9:
                        best_cost = cand_cost
                        best_move = (new_route, cand_cost)

        # Apply Move
        if best_move:
            best.route = best_move[0]
            best.calObjective() # Cập nhật lại toàn bộ thuộc tính bên trong object
            improved = True
            no_improve = 0
        else:
            no_improve += 1
            
    return best

# ------------------------------------------------
# Local Search First Improvement (Random Order)
# ------------------------------------------------
def local_search_softTW_first_improvement(ind, problem, max_iter=1000, visit_tabu_size=200):
    best = ind.copy()
    best.calObjective()
    
    n = problem.num_request
    iter_count = 0
    recent_routes = deque(maxlen=visit_tabu_size)
    recent_routes.append(tuple(best.route))

    while True:
        if max_iter is not None and iter_count >= max_iter:
            break
        iter_count += 1

        # 1. Tính toán trạng thái hiện tại
        _, _, departures, _, lateness, _, wait, _ = best.compute_route_forward(best.route, problem)
        prefix_departures = departures
        prefix_lateness = lateness
        prefix_wait = wait
        
        current_best_cost = best.objective
        improved = False 

        # Helper
        def get_prefix_info(start_idx_):
            if start_idx_ == 0:
                return 0, 0, 0.0, 0.0
            return (best.route[start_idx_-1], 
                    prefix_departures[start_idx_-1], 
                    sum(prefix_lateness[:start_idx_]),
                    sum(prefix_wait[:start_idx_]))

        order = random.sample(range(4), 4)

        for op_code in order:
            # --- RELOCATE ---
            if op_code == 0:
                for i in range(n):
                    for j in range(n+1):
                        if i == j or i == j-1: continue
                        
                        rt = best.route
                        node = rt[i]
                        temp_route = rt[:i] + rt[i+1:]
                        if j <= i:
                            new_route = temp_route[:j] + [node] + temp_route[j:]
                            start_idx = j
                        else:
                            new_route = temp_route[:j-1] + [node] + temp_route[j-1:]
                            start_idx = i

                        if tuple(new_route) in recent_routes: continue

                        prev_node, prefix_time, prefix_lat_sum, prefix_wait_sum = get_prefix_info(start_idx)
                        _, _, suf_final_time, _, suf_total_lat, _, suf_total_wait = recompute_suffix(new_route, problem, start_idx, prev_node, prefix_time)
                        
                        cand_cost = calculate_route_cost_from_metrics(suf_final_time, prefix_lat_sum + suf_total_lat, prefix_wait_sum + suf_total_wait, problem)

                        if cand_cost < current_best_cost - 1e-9:
                            best.route = new_route
                            best.calObjective()
                            recent_routes.append(tuple(best.route))
                            improved = True
                            break 
                    if improved: break

            # --- SWAP ---
            elif op_code == 1:
                for i in range(n-1):
                    for j in range(i+1, n):
                        if best.route[i] == best.route[j]: continue
                        new_route = best.route[:]
                        new_route[i], new_route[j] = new_route[j], new_route[i]

                        if tuple(new_route) in recent_routes: continue
                        start_idx = i

                        prev_node, prefix_time, prefix_lat_sum, prefix_wait_sum = get_prefix_info(start_idx)
                        _, _, suf_final_time, _, suf_total_lat, _, suf_total_wait = recompute_suffix(new_route, problem, start_idx, prev_node, prefix_time)
                        
                        cand_cost = calculate_route_cost_from_metrics(suf_final_time, prefix_lat_sum + suf_total_lat, prefix_wait_sum + suf_total_wait, problem)

                        if cand_cost < current_best_cost - 1e-9:
                            best.route = new_route
                            best.calObjective()
                            recent_routes.append(tuple(best.route))
                            improved = True
                            break
                    if improved: break

            # --- 2-OPT ---
            elif op_code == 2:
                for i in range(n-1):
                    for j in range(i+1, n):
                        rt = best.route
                        new_route = rt[:i] + rt[i:j+1][::-1] + rt[j+1:]

                        if tuple(new_route) in recent_routes: continue
                        start_idx = i

                        prev_node, prefix_time, prefix_lat_sum, prefix_wait_sum = get_prefix_info(start_idx)
                        _, _, suf_final_time, _, suf_total_lat, _, suf_total_wait = recompute_suffix(new_route, problem, start_idx, prev_node, prefix_time)
                        
                        cand_cost = calculate_route_cost_from_metrics(suf_final_time, prefix_lat_sum + suf_total_lat, prefix_wait_sum + suf_total_wait, problem)

                        if cand_cost < current_best_cost - 1e-9:
                            best.route = new_route
                            best.calObjective()
                            recent_routes.append(tuple(best.route))
                            improved = True
                            break
                    if improved: break

            # --- OR-OPT ---
            elif op_code == 3:
                k = 2
                if n >= k:
                    for i in range(n-k+1):
                        for j in range(n+1):
                            if j >= i and j <= i+k: continue
                            rt = best.route
                            block = rt[i:i+k]
                            temp_route = rt[:i] + rt[i+k:]
                            if j <= i:
                                new_route = temp_route[:j] + block + temp_route[j:]
                                start_idx = j
                            else:
                                new_route = temp_route[:j-k] + block + temp_route[j-k:]
                                start_idx = i

                            if tuple(new_route) in recent_routes: continue

                            prev_node, prefix_time, prefix_lat_sum, prefix_wait_sum = get_prefix_info(start_idx)
                            _, _, suf_final_time, _, suf_total_lat, _, suf_total_wait = recompute_suffix(new_route, problem, start_idx, prev_node, prefix_time)
                            
                            cand_cost = calculate_route_cost_from_metrics(suf_final_time, prefix_lat_sum + suf_total_lat, prefix_wait_sum + suf_total_wait, problem)

                            if cand_cost < current_best_cost - 1e-9:
                                best.route = new_route
                                best.calObjective()
                                recent_routes.append(tuple(best.route))
                                improved = True
                                break
                        if improved: break

            if improved: break

        if improved:
            continue
        else:
            break

    return best