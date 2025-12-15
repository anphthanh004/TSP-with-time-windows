import math
import random
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 1. Cấu trúc dữ liệu và Hàm đọc file
# ---------------------------------------------------------

class City:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        # Tính khoảng cách Euclidean
        distance = math.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return str(self.id)

def load_data(filename):
    cities = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # Bỏ qua các dòng chú thích hoặc dòng trống
                line = line.strip()
                if not line or line.startswith('[') or line.startswith('#'):
                    continue
                
                parts = line.split()
                # Dữ liệu mong đợi: ID X Y (ví dụ: 1 15 882)
                if len(parts) >= 3:
                    try:
                        c_id = int(parts[0])
                        c_x = float(parts[1])
                        c_y = float(parts[2])
                        cities.append(City(c_id, c_x, c_y))
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {filename}")
        return []
    return cities

# ---------------------------------------------------------
# 2. Fitness & Khởi tạo
# ---------------------------------------------------------

class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0] # Quay về điểm xuất phát
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            # Fitness càng lớn càng tốt, nên lấy nghịch đảo của distance
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

def createRoute(cityList):
    # Tạo một lộ trình ngẫu nhiên
    route = random.sample(cityList, len(cityList))
    return route

def initialPopulation(popSize, cityList):
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    # Sắp xếp fitness giảm dần (tốt nhất lên đầu)
    return sorted(fitnessResults.items(), key=lambda item: item[1], reverse=True)

# ---------------------------------------------------------
# 3. Các toán tử GA (Selection, Crossover, Mutation)
# ---------------------------------------------------------

def selection(popRanked, eliteSize):
    # Sử dụng kết hợp Elitism và Roulette Wheel Selection
    selectionResults = []
    
    # 1. Elitism: Giữ lại top eliteSize cá thể tốt nhất
    df = list(np.array(popRanked)[:, 0]) # Lấy index
    for i in range(0, eliteSize):
        selectionResults.append(df[i])
    
    # 2. Chọn lọc cho phần còn lại
    # Tính trọng số tích lũy
    fitness_values = [item[1] for item in popRanked]
    sum_fitness = sum(fitness_values)
    cum_probs = np.cumsum([f/sum_fitness for f in fitness_values])
    
    for i in range(0, len(popRanked) - eliteSize):
        pick = random.random()
        for j in range(len(cum_probs)):
            if pick <= cum_probs[j]:
                selectionResults.append(popRanked[j][0])
                break
    return selectionResults

def matingPool(population, selectionResults):
    pool = []
    for i in range(0, len(selectionResults)):
        index = int(selectionResults[i])
        pool.append(population[index])
    return pool

# Lai ghép thứ tự (Ordered Crossover - OX) - Tốt nhất cho TSP
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    # Lấy đoạn gen từ cha 1
    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    # Điền các gen còn lại từ cha 2 (giữ đúng thứ tự)
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    # Giữ lại ưu tú
    for i in range(0, eliteSize):
        children.append(matingpool[i])
    
    # Lai ghép tạo con mới
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

# Đột biến đảo ngược (Inversion Mutation)
def mutate(individual, mutationRate):
    for i in range(len(individual)):
        if(random.random() < mutationRate):
            # Chọn 2 điểm và đảo ngược đoạn giữa chúng
            idx1 = int(random.random() * len(individual))
            idx2 = int(random.random() * len(individual))
            
            start = min(idx1, idx2)
            end = max(idx1, idx2)
            
            individual[start:end+1] = individual[start:end+1][::-1]
    return individual

def mutatePopulation(population, mutationRate):
    mutatedPop = []
    for ind in population:
        mutatedInd = mutate(ind, mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

# ---------------------------------------------------------
# 4. Main Algorithm Loop
# ---------------------------------------------------------

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

def geneticAlgorithm(filename, popSize, eliteSize, mutationRate, generations):
    # 1. Load Data
    cityList = load_data(filename)
    if not cityList:
        return
    
    print(f"Đã load {len(cityList)} thành phố.")
    
    # 2. Khởi tạo quần thể
    pop = initialPopulation(popSize, cityList)
    
    print(f"Khoảng cách ban đầu: {1 / rankRoutes(pop)[0][1]:.2f}")
    
    progress = []
    initial_dist = 1 / rankRoutes(pop)[0][1]
    progress.append(initial_dist)
    
    # 3. Chạy các thế hệ
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        
        # Lấy kết quả tốt nhất hiện tại
        best_fitness = rankRoutes(pop)[0][1]
        dist = 1 / best_fitness
        progress.append(dist)
        
        if i % 50 == 0:
            print(f"Gen {i}: Khoảng cách = {dist:.2f}")

    # 4. Kết quả cuối cùng
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    min_dist = 1 / rankRoutes(pop)[0][1]
    
    print("\n-------------------------------------------")
    print(f"Khoảng cách tối ưu tìm được: {min_dist:.2f}")
    
    # Vẽ biểu đồ tiến độ
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.title('Tiến độ hội tụ GA')
    
    # Vẽ lộ trình
    x_coords = [city.x for city in bestRoute] + [bestRoute[0].x]
    y_coords = [city.y for city in bestRoute] + [bestRoute[0].y]
    
    plt.subplot(1, 2, 2)
    plt.plot(x_coords, y_coords, 'o-r', markersize=4, linewidth=1)
    plt.title(f'Lộ trình tối ưu (Dist: {min_dist:.0f})')
    plt.show()

    return bestRoute

# ---------------------------------------------------------
# 5. Chạy chương trình
# ---------------------------------------------------------

if __name__ == "__main__":
    # Cấu hình tham số GA
    FILE_NAME = "tsp200_1.txt" # Đảm bảo file này nằm cùng thư mục
    POP_SIZE = 100             # Kích thước quần thể
    ELITE_SIZE = 20            # Số lượng cá thể ưu tú giữ lại
    MUTATION_RATE = 0.01       # Tỷ lệ đột biến
    GENERATIONS = 500          # Số thế hệ
    
    geneticAlgorithm(FILE_NAME, POP_SIZE, ELITE_SIZE, MUTATION_RATE, GENERATIONS)