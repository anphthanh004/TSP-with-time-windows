import random
from .moo_structures import Individual, Problem
# -----------------------------------
# individual - greedy intialization
# -----------------------------------
def gen_route_greedy(problem, search_size=2, greedy_type ='travel_time'):
    """
    Tạo lộ trình tham lam có yếu tố ngẫu nhiên(beam search): 
        Tại mỗi bước, chọn (top k) request chưa thăm mà gần nhất và thỏa mãn khung thời gian
          - Nếu k=1: mọi individual được tạo giống nhau => mất tính đa dạng
          - Nếu k càng lớn: càng tiến về ngẫu nhiên
        Nếu không có request nào thỏa mãn, chọn ngẫu nhiên một request trong số (top k) request gần nhất
    """
    num_request = problem.num_request
    locs = list(range(1,num_request+1)) # locs là các request chưa thăm
    # locs là tên request theo 1-based, request có id là 0-based
    # vì ban đầu loc là 0 nên các request bắt đầu từ 1
    current_loc = 0
    current_time = 0
    route = []
    
    
    while locs:
        selected_loc = None
        # best_travel_time = float('inf')
        feasible_locs = []
        
        # Bước 1: Tìm các request khả thi (có thể đến kịp):
        for loc in locs:
            travel_time = problem.time_matrix[current_loc][loc]
            arrival_time = current_time + travel_time
            e, l, d = problem.request[loc-1]
            wait_time = max(0, e - arrival_time)
            if arrival_time <= l:  # Khả thi
                # feasible_locs.append((loc, travel_time, arrival_time, e, d))
                feasible_locs.append({
                    "id": loc,
                    "travel_time": travel_time,
                    "arrival_time": arrival_time,
                    "e": e, "l": l, "d": d,
                    "wait": wait_time,
                    
                })
        # Bước 2: Chọn request tốt nhất 
        # Nếu có nhiều request thỏa mãn time window, chọn cái gần nhất trong đó
        if feasible_locs:
            # sắp xếp thời gian di chuyển tăng dần
            if greedy_type == 'travel_time':
                feasible_locs.sort(key = lambda x : x['travel_time'])
            # feasible_locs.sort(key = lambda x : x[1])
            elif greedy_type == 'wait_time':
                feasible_locs.sort(key = lambda x : x['wait'])
            elif greedy_type == 'earliest':
                feasible_locs.sort(key = lambda x : x['e'])
            elif greedy_type == 'most_urgent':
                # feasible_locs.sort(key = lambda x : ((x['l']-(x['arrival_time']))/x['l']))
                feasible_locs.sort(key = lambda x : x['l']-x['arrival_time'])
            # top_k = feasible_locs[:k] if len(feasible_locs) >= k else feasible_locs
            top_k = feasible_locs[:min(len(feasible_locs), search_size)]
            selected_item = random.choice(top_k)
            selected_loc = selected_item['id']
            arrival = selected_item['arrival_time']
            e = selected_item['e']
            d = selected_item['d']
            # Cập nhật thời gian hiện tại
            current_time = max(arrival, e) + d
         
        else:
            # Nếu không có request nào thỏa mãn (đều bị trễ)
            # buộc phải chọn một cái để tiếp tục phục vụ -> chọn cái gần nhất để giảm phạt 
            # hoặc chọn một cái có l lớn nhất vì Phạt = trễ * penalty -> l lớn hơn thì phạt ít hơn
            local_locs = sorted(locs, key = lambda x : problem.time_matrix[current_loc][x])
            top_k = local_locs[:min(len(local_locs), search_size)]
            selected_item = random.choice(top_k)
            selected_loc = selected_item
            travel_time = problem.time_matrix[current_loc][selected_loc]
            arrival = current_time + travel_time
            e, l, d = problem.request[selected_loc-1]
            current_time = max(arrival, e) + d
        
        route.append(selected_loc)
        locs.remove(selected_loc)
        current_loc = selected_loc
    
    return route

def gen_route_random(num_request):
    route = random.sample(range(1,num_request+1), num_request)
    return route

def gen_pop(problem, greedy_rate=0.2, search_size=2, pop_size=100):
    pop = []
    pop_greedy = int(pop_size*greedy_rate)
    #beam search
    search_size = max(search_size, int(problem.num_request * 0.01))
    # search_size = 2
    # search_size = 1 # nếu search_size bằng 1 thì sẽ giải luôn được bài toán
    
    while len(pop) < pop_greedy:
        ind1, ind2, ind3, ind4 = [Individual(problem) for _ in range(4)]
        ind1.route = gen_route_greedy(problem, search_size, greedy_type='travel_time')
        ind2.route = gen_route_greedy(problem, search_size, greedy_type='most_urgent')
        ind3.route = gen_route_greedy(problem, search_size, greedy_type='wait_time')
        ind4.route = gen_route_greedy(problem, search_size, greedy_type='earliest')
        pop.extend([ind1, ind2, ind3, ind4])
    pop[:pop_greedy]
    while len(pop) < pop_size:
        ind = Individual(problem)
        ind.route = gen_route_random(problem.num_request)
        pop.append(ind)
        
    return pop[:pop_size]

def gen_pop_greedy1(problem, greedy_rate=0.2, search_size=2, pop_size=100):
    pop = []
    pop_greedy = int(pop_size*greedy_rate)
    #beam search
    search_size = max(search_size, int(problem.num_request * 0.01))
    
    while len(pop) < pop_greedy:
        ind = Individual(problem)
        ind.route = gen_route_greedy(problem, search_size, greedy_type='travel_time')
        pop.append(ind)
        
    while len(pop) < pop_size:
        ind = Individual(problem)
        ind.route = gen_route_random(problem.num_request)
        pop.append(ind)
        
    return pop[:pop_size]

def gen_pop_greedy2(problem, greedy_rate=0.2, search_size=2, pop_size=100):
    pop = []
    pop_greedy = int(pop_size*greedy_rate)
    #beam search
    search_size = max(search_size, int(problem.num_request * 0.01))
    
    while len(pop) < pop_greedy:
        ind = Individual(problem)
        ind.route = gen_route_greedy(problem, search_size, greedy_type='most_urgent')
        pop.append(ind)
        
    while len(pop) < pop_size:
        ind = Individual(problem)
        ind.route = gen_route_random(problem.num_request)
        pop.append(ind)
        
    return pop[:pop_size]

def gen_pop_greedy3(problem, greedy_rate=0.2, search_size=2, pop_size=100):
    pop = []
    pop_greedy = int(pop_size*greedy_rate)
    #beam search
    search_size = max(search_size, int(problem.num_request * 0.01))
    
    while len(pop) < pop_greedy:
        ind = Individual(problem)
        ind.route = gen_route_greedy(problem, search_size, greedy_type='wait_time')
        pop.append(ind)
        
    while len(pop) < pop_size:
        ind = Individual(problem)
        ind.route = gen_route_random(problem.num_request)
        pop.append(ind)
        
    return pop[:pop_size]

def gen_pop_greedy4(problem, greedy_rate=0.2, search_size=2, pop_size=100):
    pop = []
    pop_greedy = int(pop_size*greedy_rate)
    #beam search
    search_size = max(search_size, int(problem.num_request * 0.01))
    
    while len(pop) < pop_greedy:
        ind = Individual(problem)
        ind.route = gen_route_greedy(problem, search_size, greedy_type='earliest')
        pop.append(ind)
        
    while len(pop) < pop_size:
        ind = Individual(problem)
        ind.route = gen_route_random(problem.num_request)
        pop.append(ind)
        
    return pop[:pop_size]

def gen_pop_fully_random(problem, pop_size=100):
    pop = []
    while len(pop) < pop_size:
        ind = Individual(problem)
        ind.route = gen_route_random(problem.num_request)
        pop.append(ind)

    return pop