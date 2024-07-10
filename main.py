import numpy as np
import itertools


# 거리 기반 손실 함수 정의
def distance_loss(city_grid, importance_matrix):
    city_x, city_y = city_grid.shape

    corners = [(0, 0), (0, city_y - 1), (city_x - 1, 0), (city_x - 1, city_y - 1)]

    total_loss = 0.0

    for corner in corners:
        distances = np.sqrt(
            (np.arange(city_x)[:, None] - corner[0]) ** 2 + (np.arange(city_y)[None, :] - corner[1]) ** 2)
        building_distances = np.where(city_grid == 1, distances, 0.0)
        total_loss += np.sum(building_distances * importance_matrix)

    return total_loss


# 도시 크기 및 건물 개수 설정
cityX = 10
cityY = 10
building_count = 1
city = [
    {"importance": 0.8}
]

# 초기 중요도 매트릭스 설정
importance_matrix = np.zeros((cityX, cityY), dtype=np.float32)
for i, building in enumerate(city):
    importance_matrix[i // cityY, i % cityY] = building["importance"]

# 모든 가능한 위치 조합 생성
all_positions = list(itertools.product(range(cityX), range(cityY)))
possible_building_positions = list(itertools.combinations(all_positions, building_count))

# 최적의 배치를 찾기 위해 모든 조합을 평가
min_loss = float('inf')
best_city_grid = None

for positions in possible_building_positions:
    city_grid = np.zeros((cityX, cityY), dtype=np.float32)
    for pos in positions:
        city_grid[pos] = 1
    loss = distance_loss(city_grid, importance_matrix)
    if loss < min_loss:
        min_loss = loss
        best_city_grid = city_grid

# 최종 도시 구조 출력
print("최종 도시 구조:")
print(best_city_grid.astype(int))