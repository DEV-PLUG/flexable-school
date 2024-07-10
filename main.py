import tensorflow as tf
import numpy as np


# 기본적인 탄소 배출량 계산 함수 정의
def carbon_emission(city_grid, building_count, popularity):
    building_base_emission = 100.0  # 예시 기본 탄소 배출량
    transport_emission_per_person = 5.0  # 이동에 필요한 탄소 배출량
    surrounding_emission = 10.0  # 주변 상권 이동 탄소 배출량
    public_transport_reduction = 0.2  # 대중교통 사용으로 인한 탄소 배출 감소 비율
    green_area_reduction = 50.0  # 녹지의 탄소 감소량

    building_positions = tf.where(city_grid == 0)
    road_positions = tf.where(city_grid == 1)
    green_positions = tf.where(city_grid == 2)

    # 건물 탄소 배출량 계산
    building_emission = building_base_emission * building_count
    transport_emission = transport_emission_per_person * popularity
    surrounding_emission_total = surrounding_emission * tf.cast(tf.size(building_positions), tf.float32)
    public_transport_reduction_total = public_transport_reduction * transport_emission
    green_area_reduction_total = green_area_reduction * tf.cast(tf.size(green_positions), tf.float32)

    total_emission = (building_emission + transport_emission + surrounding_emission_total
                      - public_transport_reduction_total - green_area_reduction_total)

    return total_emission


# 도시 크기 및 건물 개수 설정
city_size = (4, 4)  # 도시의 크기
building_count = 5  # 건물 개수
popularity = 100  # 인기도

# 초기 도시 구조 생성 (모든 칸을 도로(1)로 초기화)
city_grid = np.ones(city_size)

# 변수로 사용할 도시 구조를 텐서플로우 변수로 변환
city_grid_var = tf.Variable(city_grid, dtype=tf.float32)

# 옵티마이저 설정 (레거시 Keras 옵티마이저 사용)
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.1)


# 손실 함수 정의
def loss_function():
    city_grid_int = tf.cast(tf.round(city_grid_var), tf.int32)  # 소수점 반올림하여 정수로 변환
    loss = carbon_emission(city_grid_int, building_count, popularity)
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
epochs = 500
for epoch in range(epochs):
    loss = train_step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")

# 최종 도시 구조 출력
final_city_grid = tf.cast(tf.round(city_grid_var), tf.int32).numpy()
print("최종 도시 구조:")
print(final_city_grid)

# import tensorflow as tf
#
# # 목표 함수 정의
# def f(x, y, z):
#     return (x - 3)**2 + (y - 2)**2 + (z - 1)**2
#
# # 변수 초기화
# x = tf.Variable(0.0)
# y = tf.Variable(0.0)
# z = tf.Variable(0.0)
#
# # 옵티마이저 설정
# optimizer = tf.optimizers.SGD(learning_rate=0.1)
#
# # 훈련 단계 정의
# @tf.function
# def train_step():
#     with tf.GradientTape() as tape:
#         loss = f(x, y, z)
#     gradients = tape.gradient(loss, [x, y, z])
#     optimizer.apply_gradients(zip(gradients, [x, y, z]))
#     return loss
#
# # 경사하강법 실행
# epochs = 500
# for epoch in range(epochs):
#     loss = train_step()
#     print(f"Epoch {epoch + 1}: x = {x.numpy()}, y = {y.numpy()}, z = {z.numpy()}, f(x, y, z) = {loss.numpy()}")
#
# print(f"최적의 x 값: {x.numpy()}")
# print(f"최적의 y 값: {y.numpy()}")
# print(f"최적의 z 값: {z.numpy()}")