import random
from .gp_structures import NodeGP, TERMINAL_SET, FUNC_SET

def cal_std_fitness(pop):
    for indi in pop:
        indi.calObjective()
        indi.fitness = indi.objective

def make_random_tree(max_depth, penalty, grow=False):
    if max_depth == 1 or (grow and random.random() < 0.4):
        return NodeGP(terminal=random.choice(TERMINAL_SET), penalty=penalty)
    
    op = random.choice(FUNC_SET)
    left = make_random_tree(max_depth-1, penalty, grow=True)
    right = make_random_tree(max_depth-1, penalty, grow=True)
    return NodeGP(op=op, left=left, right=right, penalty=penalty)

def count_nodes(node):
    if node is None: return 0
    return 1 + count_nodes(node.left) + count_nodes(node.right)

def get_node_at_index(node, target_idx, current_idx=0):
    """Trả về node tại chỉ số index (theo duyệt pre-order) và index tiếp theo"""
    if current_idx == target_idx:
        return node, current_idx + 1
    
    current_idx += 1
    if node.left:
        found, new_idx = get_node_at_index(node.left, target_idx, current_idx)
        if found: return found, new_idx
        current_idx = new_idx
    
    if node.right:
        found, new_idx = get_node_at_index(node.right, target_idx, current_idx)
        if found: return found, new_idx
        current_idx = new_idx
        
    return None, current_idx

def replace_node_at_index(root, target_idx, new_subtree, current_idx=0):
    """Tạo ra một bản sao của cây với node tại target_idx được thay thế"""
    if current_idx == target_idx:
        # return new_subtree.deepcopy(), current_idx + 1 
        return new_subtree.copy(), current_idx + 1 
    
    new_node = NodeGP(op=root.op, terminal=root.terminal, penalty=root.penalty)
    
    current_idx += 1
    
    if root.left:
        if current_idx <= target_idx: 
             size_left = count_nodes(root.left)
             if target_idx < current_idx + size_left:
                new_left, new_idx = replace_node_at_index(root.left, target_idx, new_subtree, current_idx)
                new_node.left = new_left
                #  new_node.right = root.right.deepcopy() if root.right else None
                new_node.right = root.right.copy() if root.right else None
                return new_node, new_idx
             else:
                # new_node.left = root.left.deepcopy()
                new_node.left = root.left.copy()
                current_idx += size_left
    
    if root.right:
         new_right, new_idx = replace_node_at_index(root.right, target_idx, new_subtree, current_idx)
         new_node.right = new_right
         return new_node, new_idx
         
    return new_node, current_idx

def gp_crossover(parent1, parent2, max_depth=6):
    """Lai ghép Subtree giữa 2 cá thể"""
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    tree1 = child1.tree
    tree2 = child2.tree
 
    # Đếm số node
    size1 = count_nodes(tree1)
    size2 = count_nodes(tree2)
    
    # Chọn điểm cắt ngẫu nhiên
    idx1 = random.randint(0, size1 - 1)
    idx2 = random.randint(0, size2 - 1)
    
    # while True:
    for _ in range(10):
        # Lấy subtree tại điểm cắt
        subtree1, _ = get_node_at_index(tree1, idx1)
        subtree2, _ = get_node_at_index(tree2, idx2)
        
        # Kiểm tra depth limit
        if (tree1.depth() - subtree1.depth() + subtree2.depth() <= max_depth) and \
        (tree2.depth() - subtree2.depth() + subtree1.depth() <= max_depth):
            child1.tree, _ = replace_node_at_index(child1.tree, idx1, subtree2)
            child2.tree, _ = replace_node_at_index(child2.tree, idx2, subtree1)
            child1.route = []
            child2.route = []
            return child1, child2
        # break

    return parent1, parent2

def gp_mutation(individual, max_depth=6):
    """Đột biến Subtree có kiểm soát độ sâu"""
    child = individual.copy()
    
    target_tree = child.tree
        
    size = count_nodes(target_tree)

    for _ in range(10):
        idx = random.randint(0, size - 1)
        
        # Tạo cây con đột biến (độ sâu nhỏ)
        mutation_subtree = make_random_tree(max_depth=random.randint(1, 3), grow=True, penalty=individual.problem.penalty)
        new_tree, _ = replace_node_at_index(target_tree, idx, mutation_subtree)
        
        if new_tree.depth() <= max_depth:
            child.tree = new_tree
            child.route = []
            return child
            
    # Nếu thử nhiều lần mà vẫn vi phạm độ sâu, trả về cá thể gốc (không đột biến)
    return individual