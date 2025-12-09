def simulate_tsptw(ind):
    """
    Dùng cây GP của cá thể để xây dựng lộ trình TSPTW.
    Phenotype Mapper - chuyển đổi từ Cây GP sang Lộ trình thực tế).
    Mục tiêu:
        - Đơn mục tiêu: Minimize Total Time (nếu đến trễ bị phạt)
    """
    problem = ind.problem
    n = problem.num_request
    unvisited = set(range(1, n + 1)) # Các node khách hàng (1..N)
    current_node = 0 
    current_time = 0.0
    
    route = []
    # Vòng lặp xây dựng lộ trình
    while unvisited:
        candidates = []
        
        for next_node in unvisited:
            e, l, d = problem.request[next_node-1]
            
            travel_time = problem.time_matrix[current_node][next_node]
            arrival_time = current_time + travel_time
            wait = max(0.0, e - arrival_time)
            departure = max(arrival_time, e) + d
            slack = l - arrival_time
            
            
            # 2. Dùng cây GP đánh giá độ ưu tiên
            # Lưu ý: Ta quy ước GP trả về giá trị càng NHỎ càng ưu tiên (Minimization)
            priority = ind.tree.evaluate(travel_time, e, l, wait, slack)
            
            candidates.append((priority, next_node, departure))
        
        # Chọn ứng viên có priority lớn nhất
        best_cand = min(candidates, key=lambda x: x[0])
        
        _, selected_node, finish_time = best_cand
        
        # Cập nhật trạng thái
        current_node = selected_node
        current_time = finish_time
        route.append(selected_node)
        unvisited.remove(selected_node)  
            

    ind.route = route
    
    return route


    
