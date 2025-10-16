# import pickle


# with open('/media/livanoff/Новый том2/PycharmProjects/nuscenes/nuscenes_infos_temporal_val.pkl', 'rb') as f:
#     data = pickle.load(f)
#     print()

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Путь к изображению
# image_path = '/media/livanoff/Новый том2/PycharmProjects/nuscenes/cvt_labels_nuscenes_v2/scene-0015/bev_f405bf5322cb40d3811fcbe802c0eb8d.png'

# # Чтение изображения с помощью Pillow
# img = Image.open(image_path)

# # Преобразование изображения в массив NumPy
# img_array = np.array(img)

# print(img_array.shape)

# # Визуализация изображения с помощью matplotlib
# plt.imshow(img_array)
# plt.axis('off')  # Убираем оси
# plt.show()

file_path = '/media/livanoff/Новый том2/PycharmProjects/nuscenes/cvt_labels_nuscenes_v2/scene-0065/aux_7c876ea78f93425dbda4043635da911f.npz'

# Загрузка данных из .npz
data = np.load(file_path)
array_name = data.files[0]
array_data = data[array_name]

# Проверяем форму массива (200, 200, 8)
print("Shape of array:", array_data.shape)

# Визуализация 8 изображений (по одному для каждого среза)
fig, axes = plt.subplots(2, 4, figsize=(12, 6))  # 2 строки и 4 столбца для 8 изображений
axes = axes.ravel()  # Преобразуем axes в одномерный массив для удобства

for i in range(8):
    ax = axes[i]
    ax.imshow(array_data[:, :, i])  # Отображаем срез i
    ax.axis('off')  # Убираем оси

plt.tight_layout()  # Для плотной компоновки
plt.show()
