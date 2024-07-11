import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Input data
inputs = [
    {"x": 1, "y": 1, "n": 1, "importance": [1.0]},
    {"x": 2, "y": 7, "n": 1, "importance": [0.4, 0.3]},
    {"x": 6, "y": 6, "n": 1, "importance": [0.6]}
]

# Output data in the required format
outputs = [
    [[1, 1]],         # For importance [1.0]
    [[2, 3], [2,4]],         # For importance [0.4]
    [[6, 4]]          # For importance [0.6]
]

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

# Train a decision tree regressor for predicting coordinates
model_x = DecisionTreeRegressor()
model_y = DecisionTreeRegressor()

model_x.fit(X, y[:, 0])
model_y.fit(X, y[:, 1])

# Prediction for the new input
new_input = {"x": 9, "y": 2, "n": 8, "importance": [0.4, 0.3, 0.8, 0.4, 0.3, 0.4, 0.2, 0.8]}
new_X = []

for imp in new_input['importance']:
    new_X.append([new_input['x'], new_input['y'], new_input['n'], imp])

new_X = np.array(new_X)

# Predict the coordinates
predicted_x = model_x.predict(new_X).round().astype(int)
predicted_y = model_y.predict(new_X).round().astype(int)

# Adjust coordinates to ensure uniqueness and stay within city limits
city_x = new_input['x']
city_y = new_input['y']
predicted_coordinates = []
used_coordinates = set()

for x, y in zip(predicted_x, predicted_y):
    x = x % city_x
    y = y % city_y
    while (x, y) in used_coordinates:
        x = (x + 1) % city_x
        if x == 0:
            y = (y + 1) % city_y
    used_coordinates.add((x, y))
    predicted_coordinates.append((x, y))

# Determine the size of the grid
max_x = max(coord[0] for coord in predicted_coordinates) + 1
max_y = max(coord[1] for coord in predicted_coordinates) + 1

# Initialize the grid
grid = np.zeros((max_y, max_x))

# Place importance values in the grid
for imp, (x, y) in zip(new_input['importance'], predicted_coordinates):
    grid[y, x] = imp

# Print the grid
for row in grid:
    print(' '.join(f"{int(cell):d}" if cell == int(cell) else f"{cell:.1f}" for cell in row))