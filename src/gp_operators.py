import random
from .gp_structures import NodeGP, Individual, TERMINAL_SET, FUNC_SET

# --------------------------
# 1. Tạo cây ngẫu nhiên
# --------------------------
def make_random_tree(max_depth, grow=False):
    if max_depth == 1 or (grow and random.random() < 0.4):
        return NodeGP(terminal=random.choice(TERMINAL_SET))
    
    op = random.choice(FUNC_SET)
    left = make_random_tree(max_depth-1, grow=True)
    right = make_random_tree(max_depth-1, grow=True)
    return NodeGP(op=op, left=left, right=right)

def create_population(problem, pop_size, max_depth=5):
    pop = []
    # Thêm một vài cá thể "thủ công" tốt (Greedy Heuristics cơ bản)
    # Ví dụ: Chỉ xét Distance (Nearest Neighbor)
    greedy_dist = NodeGP(terminal=('R', 0)) # R0 = Dist
    pop.append(Individual(greedy_dist))
    
    # Chỉ xét Due Date (Earliest Deadline First)
    greedy_due = NodeGP(terminal=('R', 2)) # R2 = Due
    pop.append(Individual(problem, greedy_due))
    
    while len(pop) < pop_size:
        # is_grow_tree = random.random() < 0.5
        if random.random() < 0.5:
            tree = make_random_tree(max_depth, grow = True)
        else:
            tree = make_random_tree(max_depth, grow = True)
        pop.append(Individual(problem, tree))
    return pop
# --------------------------
# 2. GP Crossover & Mutation
# --------------------------
def get_random_node(node, nodes_list):
    nodes_list.append(node)
    if node.left: get_random_node(node.left, nodes_list)
    if node.right: get_random_node(node.right, nodes_list)

def gp_crossover(p1, p2, max_depth=6):
    c1 = p1.copy()
    c2 = p2.copy()
    
    # Lấy danh sách node
    nodes1, nodes2 = [], []
    get_random_node(c1.tree, nodes1)
    get_random_node(c2.tree, nodes2)
    
    if not nodes1 or not nodes2: return c1, c2
    
    # Chọn điểm cắt
    n1 = random.choice(nodes1)
    n2 = random.choice(nodes2)
    
    # Swap data (đơn giản hóa việc swap con trỏ trong Python)
    n1.op, n2.op = n2.op, n1.op
    n1.terminal, n2.terminal = n2.terminal, n1.terminal
    n1.left, n2.left = n2.left, n1.left
    n1.right, n2.right = n2.right, n1.right
    
    # (Nếu muốn kiểm soát max_depth chặt chẽ hơn, cần code thêm đoạn check depth)
    return c1, c2

def gp_mutation(ind, max_depth=6):
    child = ind.deepcopy()
    nodes = []
    get_random_node(child.tree, nodes)
    
    if not nodes: return child
    
    target = random.choice(nodes)
    # Thay thế subtree tại target bằng một cây random mới
    new_subtree = make_random_tree(max_depth=2, grow=True)
    
    target.op = new_subtree.op
    target.terminal = new_subtree.terminal
    target.left = new_subtree.left
    target.right = new_subtree.right
    
    return child