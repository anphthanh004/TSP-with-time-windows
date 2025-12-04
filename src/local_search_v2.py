from .operators import calculate_fitness
import random
from collections import deque

def recompute_suffix(route, problem, start_idx, prev_node, current_time):
    """
    Chức năng: tính toán suffix (đuôi) của lộ trình sau khi move (Swap, Relocate, 2-opt, Or-Opt) được thực hiện bắt đầu ở chỉ số start_idx
    - Input:
        route: route sau khi move được áp dụng
        problem: chứa ma trận thời gian di chuyển, danh sách request, penalty
        start_idx: node đầu tiên bị ảnh hưởng, node trước đó không đổi
        prev_node: node trước start_idx (để tính thời gian di chuyển tới start_idx)
        current_time: thời điểm rời khỏi prev_node
    - Output:
        arrivals: thời điểm đến tại các node trong suffix
        departures: thời điểm rời node (arrival + service time)
        suffix_total_time: thời điểm hoàn thành route sau khi đi hết suffix và về depot
        lateness: thời gian trễ tại từng node (bằng 0 nếu không trễ, bằng *arrival-latest) nếu trễ)
        suffix_total_lateness: tổng lateness của suffix
        wait: thời gian đợi tại từng node (bằng 0 nếu đến kịp, bằng (earliest-arrival) nếu đến sớm)
        suffix _total_wait: tổng wait của suffix
    """
    n = problem.num_request
    arrivals = []
    departures = []
    lateness = []
    wait = []
    prev = prev_node
    t = current_time
    for idx in range(start_idx, n):
        node = route[idx]
        e_i, l_i, d_i = problem.request[node-1]
        travel = problem.time_matrix[prev][node]
        arrival = t + travel
        if arrival < e_i:
            arrival = e_i
        late = max(0.0, arrival - l_i)
        w = max(0.0, e_i - arrival)
        dep = arrival + d_i

        arrivals.append(arrival)
        lateness.append(late)
        wait.append(w)
        departures.append(dep)

        t = dep
        prev = node

    # quay về depot
    t += problem.time_matrix[prev][0]
    suffix_total_time = t
    suffix_total_lateness = sum(lateness)
    suffix_total_wait = sum(wait)
    return arrivals, departures, suffix_total_time, lateness, suffix_total_lateness, wait, suffix_total_wait

# def compute_prefix_suffix(prefix_time, prefix_lateness, suffix_total_time, suffix_total_lateness):
def compute_prefix_suffix(prefix_lateness, suffix_total_time, suffix_total_lateness):
    # suffix_total_time là thời điểm cuối cùng khi  hoàn thành suffix và về depot
    total_time = suffix_total_time
    total_lateness = prefix_lateness + suffix_total_lateness
    return total_time, total_lateness

def calculate_route_cost_from_metrics(total_time, total_lateness, problem):
    return total_time + problem.penalty * total_lateness

# ---------------------------------------
# Main Local Search (best-improvement)
# ---------------------------------------
def local_search_softTW_best_improvement(ind, problem, max_no_improve=5):
    """
    ind: Individual (có route và problem)
    problem: Problem (lưu dữ liệu bài toán)
    return improved Individual (hoặc original individual nếu không cải thiện)
    Chiến thuật best improvement:
        Ưu điểm: Đảm bảo mỗi bước đi đều là bước tối ưu nhất có thể tại thời điểm đó (tham lam tối đa).
        Nhược điểm: Tốn kém thời gian tính toán hơn so với chiến lược First-Improvement (gặp cái nào tốt hơn là đổi luôn).
                    Tuy nhiên, với kỹ thuật recompute_suffix, nhược điểm này đã được giảm thiểu đáng kể.
    """
    best = ind.copy()
    # calculate_fitness([best], problem)
    best.calObjective(problem)
    # best_cost = best.fitness
    best_cost = best.objective

    n = problem.num_request

    no_improve = 0
    iter_count = 0


    # main loop
    while no_improve < max_no_improve:
        iter_count += 1
        improved = False
        # compute forward full once
        arrivals, departures, total_time, lateness, total_lateness, wait, total_wait = best.compute_route_forward(best.route, problem)
        prefix_arrivals = arrivals
        prefix_departures = departures
        prefix_lateness = lateness
        prefix_total_lateness = total_lateness

        # tuple (move_type, params, new_route, new_cost, new_total_time, new_total_lateness)
        best_move = None

        # ---Relocate (remove i, insert at j)---
        # complexity O(n^2 * suffix)
        """
        Mục đích: Khắc phục trường hợp một khách hàng bị xếp sai thứ tự quá xa,
            gây ra đường đi vòng vèo hoặc bị trễ giờ (lateness) do đến quá muộn/quá sớm.
        """
        for i in range(n):
            for j in range(n+1):
                if i == j or i == j-1:
                    continue
                # build new route quickly
                rt = best.route
                node = rt[i]
                new_route = rt[:i] + rt[i+1:]

                # inserting before position j in the shortend list
                if j <= i:
                    new_route = new_route[:j] + [node] + new_route[j:]
                    start_idx = j
                else:
                # j>i
                    new_route = new_route[:j-1] + [node] + new_route[j-1:]
                    start_idx = i

                # compute prefix up to start_idx-1 from origianl best.route
                if start_idx == 0:
                    prev_node = 0
                    prefix_time = 0
                    prefix_lateness_sum = 0.0
                else:
                    prev_node = best.route[start_idx-1]
                    prefix_time = prefix_departures[start_idx-1]
                    prefix_lateness_sum = sum(prefix_lateness[:start_idx])

                # recompute suffix from start_idx on new_route
                # arrivals, departures, suffix_total_time, lateness, suffix_total_lateness, wait, suffix_total_wait
                arr_suf, dep_suf, suf_final_time, lat_suf, suf_total_lat, _, _ = recompute_suffix(new_route, problem, start_idx, prev_node, prefix_time)
                total_time_candidate = suf_final_time
                total_lateness_candidate = prefix_lateness_sum + suf_total_lat
                cand_cost = calculate_route_cost_from_metrics(total_time_candidate, total_lateness_candidate, problem)

                if cand_cost + 1e-9 < best_cost:
                    improved = True
                    best_cost = cand_cost
                    best_move = ("relocate", (i,j), new_route, cand_cost, total_time_candidate, total_lateness_candidate)

        # ---Swap(i,j)---
        """
        Mục đích: Hữu ích khi hai khách hàng nằm ở vị trí địa lý có vẻ "ngược đường" nhau trong lộ trình hiện tại,
            hoặc để cân bằng lại khung thời gian (Time Window).
        """
        for i in range(n-1):
            for j in range(i+1, n):
                rt = best.route
                if rt[i] == rt[j]:
                    continue
                new_route = rt[:]
                new_route[i], new_route[j] = new_route[j], new_route[i]
                start_idx = i
                if start_idx == 0:
                    prev_node = 0
                    prefix_time = 0
                    prefix_lateness_sum = 0.0
                else:
                    prev_node = best.route[start_idx-1]
                    prefix_time = prefix_departures[start_idx-1]
                    prefix_lateness_sum = sum(prefix_lateness[:start_idx])

                arr_suf, dep_suf, suf_final_time, lat_suf, suf_total_lat, _, _ = recompute_suffix(new_route, problem, start_idx, prev_node, prefix_time)
                total_time_candidate = suf_final_time
                total_lateness_candidate = prefix_lateness_sum + suf_total_lat
                cand_cost = calculate_route_cost_from_metrics(total_time_candidate, total_lateness_candidate, problem)

                if cand_cost + 1e-9 < best_cost:
                    best_cost = cand_cost
                    best_move = ("swap", (i,j), new_route, cand_cost, total_time_candidate, total_lateness_candidate)

        # ---2-opt (reverse i..j)---
        """
        Mục đích: Đây là toán tử mạnh nhất để khử các đường chéo cắt nhau (crossing edges) trên bản đồ 2D.
            Trong TSP, 2-opt là tiêu chuẩn để làm "mượt" đường đi.
        """
        for i in range(n-1):
            for j in range(i+1, n):
                rt = best.route
                # new_route = rt[:i] + list(reversed)
                new_route = rt[:i] + rt[i:j+1][::-1] + rt[j+1:]
                start_idx = i
                if start_idx == 0:
                    prev_node = 0
                    prefix_time = 0
                    prefix_lateness_sum = 0.0
                else:
                    prev_node = best.route[start_idx-1]
                    prefix_time = prefix_departures[start_idx-1]
                    prefix_lateness_sum = sum(prefix_lateness[:start_idx])

                arr_suf, dep_suf, suf_final_time, lat_suf, suf_total_lat, _, _ = recompute_suffix(new_route, problem, start_idx, prev_node, prefix_time)
                total_time_candidate = suf_final_time
                total_lateness_candidate = prefix_lateness_sum + suf_total_lat
                cand_cost = calculate_route_cost_from_metrics(total_time_candidate, total_lateness_candidate, problem)

                if cand_cost + 1e-9 < best_cost:
                    best_cost = cand_cost
                    best_move = ("2-opt", (i,j), new_route, cand_cost, total_time_candidate, total_lateness_candidate)

        # ---Or-Opt(2) (move block of length 2)---
        """
        Mục đích:
            Giữ gìn cấu trúc cục bộ: Đôi khi 2 điểm cạnh nhau (ví dụ B và C) rất gần nhau và nên đi cùng nhau.
                Nếu dùng Relocate tách B ra đi chỗ khác sẽ làm hỏng cấu trúc tốt này.
            Or-Opt di chuyển cả cụm [B, C] sang chỗ khác để tìm vị trí tốt hơn cho cả nhóm.
        """
        k = 2
        if n >= k:
            for i in range(n-k+1):
                for j in range(n+1):
                    if j >= i and j <= i+k:
                        continue # insertion in same place
                    rt = best.route
                    block = rt[i:i+k]
                    new_route = rt[:i] + rt[i+k:]
                    if j <= i:
                        new_route = new_route[:j] + block + new_route[j:]
                        start_idx = j
                    else:
                        new_route = new_route[:j-k] + block + new_route[j-k:]
                        start_idx = i

                    if start_idx == 0:
                        prev_node = 0
                        prefix_time = 0
                        prefix_lateness_sum = 0.0
                    else:
                        prev_node = best.route[start_idx-1]
                        prefix_time = prefix_departures[start_idx-1]
                        prefix_lateness_sum = sum(prefix_lateness[:start_idx])

                    # recompute inside the inner loop (fix indentation bug)
                    arr_suf, dep_suf, suf_final_time, lat_suf, suf_total_lat, _, _ = recompute_suffix(new_route, problem, start_idx, prev_node, prefix_time)
                    total_time_candidate = suf_final_time
                    total_lateness_candidate = prefix_lateness_sum + suf_total_lat
                    cand_cost = calculate_route_cost_from_metrics(total_time_candidate, total_lateness_candidate, problem)

                    if cand_cost + 1e-9 < best_cost:
                        best_cost = cand_cost
                        best_move = ("or-opt-2", (i,j), new_route, cand_cost, total_time_candidate, total_lateness_candidate)

        if best_move is not None:
            move_type, params, new_route, new_cost, new_total_time, new_total_lateness = best_move
            # apply to best individual
            best.route = new_route
            # recalc objective properly to keep internal metrics consistent
            best.calObjective(problem)
            improved = True
            no_improve = 0
        else:
            no_improve += 1

    return best


# ------------------------------------------------
# Local Search First Improvement (Random Order)
# ------------------------------------------------
# def local_search_softTW_first_improvement(ind, problem, max_iter=None):
def local_search_softTW_first_improvement(ind, problem, max_iter=1000, visit_tabu_size=200):
    """
    First-improvement local search with several robustness fixes:

    - default max_iter to avoid accidental infinite loop
    - use a short-term tabu (visited recent routes set) to prevent immediate cycles
    - always call best.calObjective(problem) after accepting a move to keep internal metrics consistent
    - keep the same random operator order behavior
    """
    best = ind.copy()
    best.calObjective(problem)

    n = problem.num_request
    iter_count = 0

    # short-term memory to avoid cycling between very recent routes
    recent_routes = deque(maxlen=visit_tabu_size)
    recent_routes.append(tuple(best.route))

    # Vòng lặp chính (Restart Loop)
    while True:
        if max_iter is not None and iter_count >= max_iter:
            break
        iter_count += 1

        # 1. Chuẩn bị dữ liệu cho cấu hình hiện tại
        arrivals, departures, total_time, lateness, total_lateness, wait, total_wait = best.compute_route_forward(best.route, problem)

        prefix_departures = departures
        prefix_lateness = lateness

        current_best_cost = best.objective
        improved = False  # Cờ kiểm soát

        # Helper lấy thông tin prefix
        def get_prefix_info(start_idx_):
            if start_idx_ == 0:
                return 0, 0, 0.0
            return best.route[start_idx_-1], prefix_departures[start_idx_-1], sum(prefix_lateness[:start_idx_])

        # 2. Random thứ tự các toán tử
        # 0: Relocate, 1: Swap, 2: 2-opt, 3: Or-opt
        order = random.sample(range(4), 4)

        # Duyệt qua từng loại toán tử theo thứ tự ngẫu nhiên
        for op_code in order:

            # --- RELOCATE ---
            if op_code == 0:
                for i in range(n):
                    for j in range(n+1):
                        if i == j or i == j-1:
                            continue

                        rt = best.route
                        node = rt[i]
                        temp_route = rt[:i] + rt[i+1:]
                        if j <= i:
                            new_route = temp_route[:j] + [node] + temp_route[j:]
                            start_idx = j
                        else:
                            new_route = temp_route[:j-1] + [node] + temp_route[j-1:]
                            start_idx = i

                        # skip recently visited routes to reduce cycling
                        tup = tuple(new_route)
                        if tup in recent_routes:
                            continue

                        prev_node, prefix_time, prefix_lateness_sum = get_prefix_info(start_idx)
                        _, _, suf_final_time, _, suf_total_lat, _, _ = recompute_suffix(new_route, problem, start_idx, prev_node, prefix_time)
                        cand_cost = calculate_route_cost_from_metrics(suf_final_time, prefix_lateness_sum + suf_total_lat, problem)

                        if cand_cost + 1e-9 < current_best_cost:
                            best.route = new_route
                            # recalc objective and internal metrics consistently
                            best.calObjective(problem)
                            recent_routes.append(tuple(best.route))
                            improved = True
                            break  # Break j loop
                    if improved:
                        break  # Break i loop

            # --- SWAP ---
            elif op_code == 1:
                for i in range(n-1):
                    for j in range(i+1, n):
                        if best.route[i] == best.route[j]:
                            continue

                        new_route = best.route[:]
                        new_route[i], new_route[j] = new_route[j], new_route[i]

                        # skip recently visited
                        tup = tuple(new_route)
                        if tup in recent_routes:
                            continue

                        start_idx = i

                        prev_node, prefix_time, prefix_lateness_sum = get_prefix_info(start_idx)
                        _, _, suf_final_time, _, suf_total_lat, _, _ = recompute_suffix(new_route, problem, start_idx, prev_node, prefix_time)
                        cand_cost = calculate_route_cost_from_metrics(suf_final_time, prefix_lateness_sum + suf_total_lat, problem)

                        if cand_cost + 1e-9 < current_best_cost:
                            best.route = new_route
                            best.calObjective(problem)
                            recent_routes.append(tuple(best.route))
                            improved = True
                            break  # Break j loop
                    if improved:
                        break  # Break i loop

            # --- 2-OPT ---
            elif op_code == 2:
                for i in range(n-1):
                    for j in range(i+1, n):
                        rt = best.route
                        new_route = rt[:i] + rt[i:j+1][::-1] + rt[j+1:]

                        tup = tuple(new_route)
                        if tup in recent_routes:
                            continue

                        start_idx = i

                        prev_node, prefix_time, prefix_lateness_sum = get_prefix_info(start_idx)
                        _, _, suf_final_time, _, suf_total_lat, _, _ = recompute_suffix(new_route, problem, start_idx, prev_node, prefix_time)
                        cand_cost = calculate_route_cost_from_metrics(suf_final_time, prefix_lateness_sum + suf_total_lat, problem)

                        if cand_cost + 1e-9 < current_best_cost:
                            best.route = new_route
                            best.calObjective(problem)
                            recent_routes.append(tuple(best.route))
                            improved = True
                            break  # Break j loop
                    if improved:
                        break  # Break i loop

            # --- OR-OPT ---
            elif op_code == 3:
                k = 2
                if n >= k:
                    for i in range(n-k+1):
                        for j in range(n+1):
                            if j >= i and j <= i+k:
                                continue

                            rt = best.route
                            block = rt[i:i+k]
                            temp_route = rt[:i] + rt[i+k:]
                            if j <= i:
                                new_route = temp_route[:j] + block + temp_route[j:]
                                start_idx = j
                            else:
                                new_route = temp_route[:j-k] + block + temp_route[j-k:]
                                start_idx = i

                            tup = tuple(new_route)
                            if tup in recent_routes:
                                continue

                            prev_node, prefix_time, prefix_lateness_sum = get_prefix_info(start_idx)
                            _, _, suf_final_time, _, suf_total_lat, _, _ = recompute_suffix(new_route, problem, start_idx, prev_node, prefix_time)
                            cand_cost = calculate_route_cost_from_metrics(suf_final_time, prefix_lateness_sum + suf_total_lat, problem)

                            if cand_cost + 1e-9 < current_best_cost:
                                best.route = new_route
                                best.calObjective(problem)
                                recent_routes.append(tuple(best.route))
                                improved = True
                                break  # Break j loop
                        if improved:
                            break  # Break i loop

            # --- KIỂM TRA SAU MỖI OPERATOR ---
            # Nếu đã tìm thấy cải thiện ở operator này rồi -> Break khỏi vòng lặp order
            # Để không thực hiện tiếp các operator khác trên cấu hình cũ nữa.
            if improved:
                break

        # 3. Xử lý logic vòng lặp chính
        if improved:
            # Nếu có cải thiện: Quay lại đầu vòng while True, tính lại prefix cho route MỚI
            continue
        else:
            # Nếu chạy hết cả list order (cả 4 operators) mà không tìm thấy gì -> Tối ưu cục bộ
            break

    return best