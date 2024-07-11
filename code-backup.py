import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Input data
inputs = [{'x': 1, 'y': 1, 'n': 1, 'importance': [1.0]}, {'x': 2, 'y': 7, 'n': 1, 'importance': [0.4]}, {'x': 6, 'y': 6, 'n': 1, 'importance': [0.6]}, {'x': 9, 'y': 2, 'n': 8, 'importance': [0.4, 0.3, 0.8, 0.4, 0.3, 0.4, 0.2, 0.8]}, {'x': 5, 'y': 10, 'n': 3, 'importance': [0.9, 0.3, 0.7]}, {'x': 6, 'y': 4, 'n': 7, 'importance': [0.9, 0.1, 0.4, 0.6, 0.4, 0.8, 0.7]}, {'x': 2, 'y': 9, 'n': 2, 'importance': [0.7, 0.8]}, {'x': 8, 'y': 6, 'n': 1, 'importance': [1.0]}, {'x': 8, 'y': 3, 'n': 9, 'importance': [0.3, 0.3, 0.6, 0.8, 0.8, 1.0, 1.0, 1.0, 0.9]}, {'x': 10, 'y': 2, 'n': 9, 'importance': [0.3, 0.9, 0.3, 0.5, 0.6, 1.0, 0.9, 0.7, 0.7]}, {'x': 1, 'y': 8, 'n': 7, 'importance': [0.3, 0.9, 0.5, 0.5, 0.6, 0.7, 0.3]}, {'x': 4, 'y': 5, 'n': 3, 'importance': [0.1, 0.3, 0.2]}, {'x': 5, 'y': 4, 'n': 9, 'importance': [1.0, 0.8, 0.4, 0.9, 0.1, 0.5, 0.2, 0.4, 0.9]}, {'x': 6, 'y': 4, 'n': 9, 'importance': [0.6, 0.1, 0.7, 1.0, 0.4, 0.8, 0.9, 1.0, 0.2]}, {'x': 5, 'y': 4, 'n': 1, 'importance': [0.4]}, {'x': 8, 'y': 3, 'n': 6, 'importance': [0.8, 0.4, 0.3, 0.6, 0.1, 0.6]}, {'x': 7, 'y': 3, 'n': 8, 'importance': [1.0, 0.6, 0.9, 0.9, 1.0, 0.6, 0.4, 0.8]}, {'x': 3, 'y': 2, 'n': 2, 'importance': [0.4, 0.9]}, {'x': 1, 'y': 5, 'n': 2, 'importance': [1.0, 0.8]}, {'x': 2, 'y': 9, 'n': 10, 'importance': [0.2, 0.3, 0.3, 0.1, 0.6, 0.5, 0.9, 0.4, 1.0, 1.0]}, {'x': 2, 'y': 1, 'n': 1, 'importance': [0.9]}, {'x': 1, 'y': 1, 'n': 1, 'importance': [0.5]}, {'x': 6, 'y': 4, 'n': 4, 'importance': [0.3, 0.4, 0.9, 0.9]}, {'x': 1, 'y': 3, 'n': 3, 'importance': [0.7, 1.0, 0.9]}, {'x': 7, 'y': 7, 'n': 5, 'importance': [1.0, 0.5, 0.8, 0.4, 0.8]}, {'x': 6, 'y': 1, 'n': 6, 'importance': [0.4, 0.1, 0.5, 0.1, 0.4, 0.3]}, {'x': 9, 'y': 9, 'n': 3, 'importance': [0.9, 0.6, 0.9]}, {'x': 3, 'y': 1, 'n': 1, 'importance': [0.9]}, {'x': 4, 'y': 4, 'n': 4, 'importance': [0.5, 0.8, 0.6, 0.5]}, {'x': 8, 'y': 10, 'n': 1, 'importance': [1.0]}, {'x': 4, 'y': 8, 'n': 4, 'importance': [0.9, 0.5, 1.0, 0.8]}, {'x': 2, 'y': 4, 'n': 3, 'importance': [0.9, 1.0, 0.2]}, {'x': 9, 'y': 4, 'n': 5, 'importance': [0.5, 0.6, 0.6, 0.2, 0.9]}, {'x': 9, 'y': 9, 'n': 4, 'importance': [0.1, 0.3, 0.9, 0.9]}, {'x': 10, 'y': 2, 'n': 10, 'importance': [0.2, 0.2, 0.5, 0.5, 1.0, 0.1, 1.0, 1.0, 0.9, 0.4]}, {'x': 4, 'y': 9, 'n': 3, 'importance': [0.2, 0.4, 0.9]}, {'x': 2, 'y': 3, 'n': 2, 'importance': [0.2, 0.8]}, {'x': 10, 'y': 1, 'n': 5, 'importance': [0.9, 0.9, 0.5, 0.1, 0.3]}, {'x': 6, 'y': 6, 'n': 1, 'importance': [0.8]}, {'x': 6, 'y': 4, 'n': 7, 'importance': [0.8, 0.5, 0.1, 0.6, 0.2, 0.1, 0.2]}, {'x': 6, 'y': 4, 'n': 4, 'importance': [0.3, 0.6, 0.2, 0.1]}, {'x': 1, 'y': 1, 'n': 1, 'importance': [0.7]}, {'x': 1, 'y': 2, 'n': 2, 'importance': [0.8, 0.8]}, {'x': 6, 'y': 1, 'n': 1, 'importance': [1.0]}, {'x': 6, 'y': 3, 'n': 6, 'importance': [0.1, 1.0, 0.2, 1.0, 0.4, 0.7]}, {'x': 3, 'y': 5, 'n': 5, 'importance': [0.9, 0.4, 0.3, 0.1, 1.0]}, {'x': 9, 'y': 3, 'n': 10, 'importance': [0.4, 0.7, 0.8, 0.8, 0.3, 0.2, 0.6, 0.2, 0.1, 1.0]}]

# Output data in the required format
outputs = [[[0, 0]], [[0, 3]], [[2, 3]], [[2, 0], [2, 1], [3, 0], [3, 1], [4, 0], [4, 1], [5, 0], [5, 1]], [[2, 4], [2, 5], [2, 6]], [[1, 1], [1, 2], [2, 1], [2, 2], [3, 1], [3, 2], [4, 1]], [[0, 4], [1, 4]], [[3, 2]], [[1, 1], [2, 0], [2, 1], [3, 0], [3, 1], [3, 2], [4, 1], [5, 1], [6, 1]], [[2, 0], [3, 0], [3, 1], [4, 0], [4, 1], [5, 0], [5, 1], [6, 0], [6, 1]], [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6]], [[1, 1], [1, 2], [2, 2]], [[1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [2, 3], [3, 0], [3, 1], [3, 2]], [[1, 1], [1, 2], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [4, 1], [4, 2]], [[2, 1]], [[2, 1], [3, 1], [3, 2], [4, 1], [4, 2], [5, 1]], [[1, 1], [2, 0], [2, 1], [3, 0], [3, 1], [4, 1], [4, 2], [5, 1]], [[1, 0], [1, 1]], [[0, 0], [0, 1]], [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]], [[0, 0]], [[0, 0]], [[2, 1], [2, 2], [3, 1], [3, 2]], [[0, 0], [0, 1], [0, 2]], [[2, 3], [3, 2], [3, 3], [3, 4], [4, 3]], [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]], [[3, 4], [4, 3], [4, 4]], [[0, 0]], [[1, 1], [1, 2], [2, 1], [2, 2]], [[3, 5]], [[1, 3], [1, 4], [2, 3], [2, 4]], [[0, 1], [0, 2], [1, 1]], [[3, 1], [4, 1], [4, 2], [5, 1], [5, 2]], [[3, 4], [4, 3], [4, 4], [4, 5]], [[2, 0], [2, 1], [3, 0], [3, 1], [4, 0], [4, 1], [5, 0], [5, 1], [6, 0], [6, 1]], [[1, 3], [1, 4], [2, 4]], [[0, 1], [1, 1]], [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], [[2, 3]], [[2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [4, 1], [4, 2]], [[2, 1], [2, 2], [3, 1], [3, 2]], [[0, 0]], [[0, 0], [0, 1]], [[0, 0]], [[1, 1], [2, 1], [2, 2], [3, 1], [3, 2], [4, 1]], [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2]], [[1, 1], [2, 1], [3, 1], [4, 1], [4, 2], [5, 0], [5, 1], [5, 2], [6, 0], [6, 1]]]

# Flatten the data for model training
X = []
y = []

for input_data, output_data in zip(inputs, outputs):
    x, y_start, n, importance = input_data['x'], input_data['y'], input_data['n'], input_data['importance']
    for imp, coords in zip(importance, output_data):
        X.append([x, y_start, n, imp])
        y.append(coords)

X = np.array(X)
y = np.array(y)

# 모델 구성
model_x = Sequential()
model_x.add(Dense(64, input_dim=4, activation='relu'))
model_x.add(Dense(32, activation='relu'))
model_x.add(Dense(1))

model_y = Sequential()
model_y.add(Dense(64, input_dim=4, activation='relu'))
model_y.add(Dense(32, activation='relu'))
model_y.add(Dense(1))

# 모델 컴파일
model_x.compile(optimizer='adam', loss='mean_squared_error')
model_y.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
model_x.fit(X, y[:, 0], epochs=100, verbose=1)
model_y.fit(X, y[:, 1], epochs=100, verbose=1)

# 새로운 입력에 대한 예측
new_input = {"x": 10, "y": 10, "n": 8, "importance": [0.4, 0.3, 0.8, 0.4, 0.3, 0.4, 0.2, 0.8]}
new_X = []

for imp in new_input['importance']:
    new_X.append([new_input['x'], new_input['y'], new_input['n'], imp])

new_X = np.array(new_X)

# 좌표 예측
predicted_x = model_x.predict(new_X).round().astype(int)
predicted_y = model_y.predict(new_X).round().astype(int)

# 유일성 보장 및 도시 범위 내 좌표 조정
city_x = new_input['x']
city_y = new_input['y']
predicted_coordinates = []
used_coordinates = set()

for x, y in zip(predicted_x, predicted_y):
    x = x.item() % city_x
    y = y.item() % city_y
    while (x, y) in used_coordinates:
        x = (x + 1) % city_x
        if x == 0:
            y = (y + 1) % city_y
    used_coordinates.add((x, y))
    predicted_coordinates.append((x, y))

# 그리드 크기 결정
max_x = max(coord[0] for coord in predicted_coordinates) + 1
max_y = max(coord[1] for coord in predicted_coordinates) + 1

# 그리드 초기화
grid = np.zeros((max_y, max_x))

# 그리드에 중요도 값 배치
for imp, (x, y) in zip(new_input['importance'], predicted_coordinates):
    grid[y, x] = imp

# 그리드 출력
for row in grid:
    print(' '.join(f"{int(cell):d}" if cell == int(cell) else f"{cell:.1f}" for cell in row))