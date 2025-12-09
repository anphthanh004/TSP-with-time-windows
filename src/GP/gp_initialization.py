import random
from .gp_structures import NodeGP, Individual
from .gp_operators import make_random_tree


def gen_pop(problem, pop_size, max_depth=5):
    pop = []
    greedy_dist = NodeGP(terminal=('R', 0), penalty = problem.penalty) # R0 = Dist
    pop.append(Individual(problem, greedy_dist))
    
    greedy_due = NodeGP(terminal=('R', 2), penalty = problem.penalty) # R2 = Due
    pop.append(Individual(problem, greedy_due))
    
    while len(pop) < pop_size:
        if random.random() < 0.5:
            tree = make_random_tree(max_depth, penalty=problem.penalty, grow = True)
        else:
            tree = make_random_tree(max_depth, penalty=problem.penalty, grow = False)
        pop.append(Individual(problem, tree))
    return pop