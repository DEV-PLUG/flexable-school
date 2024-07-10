import tensorflow as tf
import numpy as np


# 거리 기반 손실 함수 정의
def distance_loss(city_grid, importance_matrix):
    city_shape = tf.shape(city_grid)
    city_x, city_y = city_shape[0], city_shape[1]

    corners = [(0, 0), (0, city_y - 1), (city_x - 1, 0), (city_x - 1, city_y - 1)]

    total_loss = 0.0

    for corner in corners:
        distances = tf.sqrt(
            tf.cast(tf.square(tf.range(city_x)[:, None] - corner[0]) + tf.square(tf.range(city_y)[None, :] - corner[1]),
                    tf.float32))
        building_distances = tf.where(city_grid == 1, distances, 0.0)
        total_loss += tf.reduce_sum(building_distances * importance_matrix)

    return total_loss


# 도시 크기 및 건물 개수 설정
cityX = 3
cityY = 3
building_count = 1
city = [
    {"importance": 0.8}
]

# 초기 도시 구조 생성 (모든 칸을 도로(0)로 초기화)
city_grid = np.zeros((cityX, cityY), dtype=np.float32)
importance_matrix = np.zeros((cityX, cityY), dtype=np.float32)

# 중요도 매트릭스 설정
for i, building in enumerate(city):
    importance_matrix[i // cityY, i % cityY] = building["importance"]

# 초기 건물 배치를 무작위로 설정
initial_positions = np.random.choice(cityX * cityY, building_count, replace=False)
for pos in initial_positions:
    city_grid[pos // cityY, pos % cityY] = 1

# 변수로 사용할 도시 구조를 텐서플로우 변수로 변환
city_grid_var = tf.Variable(city_grid, dtype=tf.float32)
importance_matrix_var = tf.constant(importance_matrix, dtype=tf.float32)

# 옵티마이저 설정
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)


# 손실 함수 정의
def loss_function():
    city_grid_int = tf.cast(tf.round(city_grid_var), tf.int32)  # 소수점 반올림하여 정수로 변환
    loss = distance_loss(city_grid_int, importance_matrix_var)
    return loss


# 훈련 단계 정의
@tf.function
def train_step():
    with tf.GradientTape() as tape:
        loss = loss_function()
    gradients = tape.gradient(loss, [city_grid_var])
    # 그라디언트가 None이 아닌지 확인
    gradients = [g if g is not None else tf.zeros_like(v) for g, v in zip(gradients, [city_grid_var])]
    optimizer.apply_gradients(zip(gradients, [city_grid_var]))
    return loss


# 경사하강법 실행
epochs = 1000
for epoch in range(epochs):
    loss = train_step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")

# 최종 도시 구조 출력
final_city_grid = tf.cast(tf.round(city_grid_var), tf.int32).numpy()
print("최종 도시 구조:")
print(final_city_grid)