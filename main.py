import itertools
import math

def calculate_distance_sum(buildings, importances, x, y):
    corners = [(0, 0), (0, y - 1), (x - 1, 0), (x - 1, y - 1)]
    total_importance = 0

    for (i, j), importance in zip(buildings, importances):
        distance_sum = 0
        for corner in corners:
            distance = math.sqrt((corner[0] - i) ** 2 + (corner[1] - j) ** 2)
            distance_sum += distance
        print(distance_sum)
        total_importance += distance_sum * importance

    return total_importance

def generate_city(x, y, building_positions, importances):
    city = [[0] * y for _ in range(x)]
    for (i, j), importance in zip(building_positions, importances):
        city[i][j] = importance
    return city

def best_building_placement(n, importances, x, y):
    min_importance_sum = float('inf')
    best_city = None
    possible_positions = [(i, j) for i in range(x) for j in range(y)]
    print(possible_positions)

    for building_positions in itertools.combinations(possible_positions, n):
        importance_sum = calculate_distance_sum(building_positions, importances, x, y)
        print(importance_sum)
        if importance_sum < min_importance_sum:
            min_importance_sum = importance_sum
            best_city = generate_city(x, y, building_positions, importances)

    print(calculate_distance_sum([(1,1), (1,3)], importances, x, y))
    print(min_importance_sum)
    return best_city

# 입력 예시
n = 5
importances = [0.3, 0.5, 0.3, 0.4, 0.4]
x = 10
y = 10

best_city = best_building_placement(n, importances, x, y)
for row in best_city:
    print(row)
print()
