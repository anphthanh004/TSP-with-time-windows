import math
from .operators import calculate_fitness

def simulate_tsptw_moo(individual, problem):
    """
    Dùng cây GP của cá thể để xây dựng lộ trình TSPTW.
    Mục tiêu:
        - Đơn mục tiêu: Minimize Total Time (nếu đến trễ bị phạt)
        - MOO :1. Minimize Total Time (Makespan)
            2. Minimize Violations (Số lần đến trễ) - Hoặc coi là hard constraint
    """
    
    # Dữ liệu bài toán
    n = problem.num_request
    unvisited = set(range(1, n + 1)) # Các node khách hàng (1..N)
    current_node = 0 # Bắt đầu từ Depot
    current_time = 0.0
    
    route = [0]
    violations = 0
    
    # Vòng lặp xây dựng lộ trình
    while unvisited:
        candidates = []
        
        for next_node in unvisited:
            # 1. Tính toán các thuộc tính (Terminals)
            # Travel Time / Distance
            dist = problem.time_matrix[current_node][next_node]
            arrival_time = current_time + dist
            
            e, l, service = problem.request[next_node-1] # request 0-based
            
            # Waiting time: Nếu đến sớm hơn e -> phải chờ
            wait = max(0.0, e - arrival_time)
            
            # Thời điểm bắt đầu phục vụ thực tế
            start_service = max(arrival_time, e)
            
            # Slack time: Thời gian còn dư trước khi bị muộn
            slack = l - arrival_time
            
            # Kiểm tra trễ (Hard constraint check - optional)
            # Nếu arrival_time > l -> Violation
            is_late = arrival_time > l
            
            # 2. Dùng cây GP đánh giá độ ưu tiên
            # Lưu ý: Ta quy ước GP trả về giá trị càng NHỎ càng ưu tiên (Minimization)
            # Hoặc càng LỚN càng ưu tiên. Ở đây tôi dùng: CÀNG NHỎ CÀNG TỐT (Priority Score)
            priority = individual.tree.evaluate(dist, e, l, wait, slack)
            
            # Nếu vi phạm time window quá nặng, phạt điểm priority cực lớn
            if is_late:
                 priority += 100000.0 # Penalty
            
            candidates.append((priority, next_node, start_service + service, is_late))
        
        # Chọn ứng viên có priority nhỏ nhất
        # candidates.sort(key=lambda x: x[0]) 
        # best_cand = candidates[0] 
        # (Để nhanh hơn dùng min)
        best_cand = min(candidates, key=lambda x: x[0])
        
        _, selected_node, finish_time, late_flag = best_cand
        
        # Cập nhật trạng thái
        current_node = selected_node
        current_time = finish_time
        route.append(selected_node)
        unvisited.remove(selected_node)
        
        if late_flag:
            violations += 1
            
    # Quay về depot
    current_time += problem.time_matrix[current_node][0]
    route.append(0)
    
    # Gán kết quả vào cá thể
    individual.built_route = route
    individual.total_time = current_time
    individual.violations = violations
    
    # Tính Fitness (NSGA-II Minimize cả 2)
    # f1: Số lượng vi phạm (Violations)
    # f2: Tổng thời gian (Total Time)
    individual.fitness = (violations, current_time)
    
    return individual.fitness


def simulate_tsptw(ind, problem):
    """
    Dùng cây GP của cá thể để xây dựng lộ trình TSPTW.
    Phenotype Mapper - chuyển đổi từ Cây GP sang Lộ trình thực tế).
    Mục tiêu:
        - Đơn mục tiêu: Minimize Total Time (nếu đến trễ bị phạt)
    """
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
        best_cand = max(candidates, key=lambda x: x[0])
        
        _, selected_node, finish_time = best_cand
        
        # Cập nhật trạng thái
        current_node = selected_node
        current_time = finish_time
        route.append(selected_node)
        unvisited.remove(selected_node)  
            

    ind.route = route
    
    return route
    # ind.calObjective()
    # calculate_fitness([ind], problem)

    
