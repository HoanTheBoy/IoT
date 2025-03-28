import matplotlib.pyplot as plt
import numpy as np

def draw_neural_net(ax, left, right, bottom, top, layer_sizes, weights, biases):
    '''
    Vẽ sơ đồ mạng nơ-ron với các tầng và kết nối có trọng số.
    
    Parameters:
    - ax: Axes của Matplotlib để vẽ.
    - left, right, bottom, top: Vị trí của sơ đồ.
    - layer_sizes: Danh sách số nơ-ron ở mỗi tầng.
    - weights: Danh sách ma trận trọng số giữa các tầng.
    - biases: Danh sách vector bias cho các tầng.
    '''
    v_spacing = (top - bottom) / float(max(layer_sizes))  # Khoảng cách dọc giữa các nơ-ron
    h_spacing = (right - left) / float(len(layer_sizes) - 1)  # Khoảng cách ngang giữa các tầng
    
    # Vẽ các nơ-ron
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
            # Ghi chú cho nơ-ron
            if n == 0:
                ax.text(n * h_spacing + left, layer_top - m * v_spacing, f'Input {m+1}', 
                        ha='center', va='center')
            elif n == len(layer_sizes) - 1:
                ax.text(n * h_spacing + left, layer_top - m * v_spacing, 'Output', 
                        ha='center', va='center')
            else:
                ax.text(n * h_spacing + left, layer_top - m * v_spacing, f'H{n}{m+1}', 
                        ha='center', va='center')
    
    # Vẽ các kết nối và ghi trọng số, bias
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k')
                ax.add_artist(line)
                # Ghi trọng số
                weight = weights[n][m, o]
                ax.text((n * h_spacing + left + (n + 1) * h_spacing + left) / 2,
                        (layer_top_a - m * v_spacing + layer_top_b - o * v_spacing) / 2,
                        f'{weight}', ha='center', va='center', color='r')
        # Ghi bias cho tầng tiếp theo
        for o in range(layer_size_b):
            bias = biases[n][o]
            ax.text((n + 1) * h_spacing + left + 0.1 * h_spacing,
                    layer_top_b - o * v_spacing,
                    f'b={bias}', ha='left', va='center', color='b')

# Thiết lập tham số mạng
layer_sizes = [2, 2, 1]  # Input: 2, Hidden: 2, Output: 1
weights = [
    np.array([[1, 1], [1, 1]]).T,  # Trọng số từ input đến hidden (2x2)
    np.array([[1], [-2]])           # Trọng số từ hidden đến output (2x1)
]
biases = [
    np.array([-0.5, -1.5]),  # Bias cho hidden layer
    np.array([0])            # Bias cho output layer
]

# Tạo và vẽ sơ đồ
fig = plt.figure(figsize=(10, 6))
ax = fig.gca()
ax.axis('off')  # Tắt các trục tọa độ
draw_neural_net(ax, 0.1, 0.9, 0.1, 0.9, layer_sizes, weights, biases)
plt.show()