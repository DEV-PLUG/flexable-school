import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# 입력 데이터
inputs = [
    {
        "x": 1,
        "y": 1,
        "n": 1,
        "importance": [1.0]
    },
    {
        "x": 2,
        "y": 7,
        "n": 1,
        "importance": [0.4]
    },
    {
        "x": 6,
        "y": 6,
        "n": 1,
        "importance": [0.6]
    },
    {
        "x": 9,
        "y": 2,
        "n": 8,
        "importance": [0.4, 0.3, 0.8, 0.4, 0.3, 0.4, 0.2, 0.8]
    },
]

# 출력 데이터
outputs = [
    [
        [1.0]
    ],
    [
        [0, 0, 0, 0.4, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0.6, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ],
    [
        [0, 0],
        [0, 0],
        [0.4, 0.3],
        [0.8, 0.4],
        [0.3, 0.4],
        [0.2, 0.8],
        [0, 0],
        [0, 0],
        [0, 0]
    ],
]

# 모델 학습을 위한 데이터 변환
X = []
y = []

for input_data, output_data in zip(inputs, outputs):
    x, y_start, n, importance = input_data['x'], input_data['y'], input_data['n'], input_data['importance']
    for imp, coords_list in zip(importance, output_data):
        for coords in coords_list:
            X.append([x, y_start, n, imp])
            y.append(coords)

X = np.array(X)
y = np.array(y, dtype=object)  # ragged 배열 처리

# 결정 트리 회귀 모델 학습
model_x = DecisionTreeRegressor()
model_y = DecisionTreeRegressor()

model_x.fit(X, y[:, 0])
model_y.fit(X, y[:, 1])

# 학습 데이터에 대한 예측
predicted_x = model_x.predict(X).round().astype(int)
predicted_y = model_y.predict(X).round().astype(int)

# 실제 좌표와 예측 좌표 산점도
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y[:, 0], predicted_x, color='blue', label='예측 X 좌표')
plt.plot(y[:, 0], y[:, 0], color='red', linestyle='--', label='실제 X 좌표')
plt.xlabel('실제 X 좌표')
plt.ylabel('예측 X 좌표')
plt.legend()
plt.title('실제 X 좌표 vs 예측 X 좌표')

plt.subplot(1, 2, 2)
plt.scatter(y[:, 1], predicted_y, color='green', label='예측 Y 좌표')
plt.plot(y[:, 1], y[:, 1], color='red', linestyle='--', label='실제 Y 좌표')
plt.xlabel('실제 Y 좌표')
plt.ylabel('예측 Y 좌표')
plt.legend()
plt.title('실제 Y 좌표 vs 예측 Y 좌표')

plt.tight_layout()
plt.show()

# 결정 계수 R² 점수 출력
r2_x = model_x.score(X, y[:, 0])
r2_y = model_y.score(X, y[:, 1])
print(f"X 좌표에 대한 R² 점수: {r2_x:.2f}")
print(f"Y 좌표에 대한 R² 점수: {r2_y:.2f}")