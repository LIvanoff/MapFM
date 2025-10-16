import torch

def get_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_MB = total_bytes / (1024 ** 2)
    total_GB = total_bytes / (1024 ** 3)
    return total_params / 1e6, total_GB  # миллионы параметров, размер в ГБ

model_versions = [
    "radio_v2.5-g",  # ViT-H/14
    "radio_v2.5-h",  # ViT-H/16
    "radio_v2.5-l",  # ViT-L/16
    "radio_v2.5-b",  # ViT-B/16
    "e-radio_v2",    # E-RADIO
]

results = {}

for version in model_versions:
    print(f"Загружается модель: {version}")
    model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=version, progress=True, skip_validation=True)
    model.eval()
    with torch.no_grad():
        num_params_millions, size_gb = get_model_info(model)
        results[version] = (num_params_millions, size_gb)
        print(f"→ {version}: {num_params_millions:.2f}M параметров, {size_gb:.2f} ГБ\n")

# Вывод всех результатов
print("Итоговая таблица:")
for version, (params, size) in results.items():
    print(f"{version:15s} | {params:8.2f}M параметров | {size:6.2f} ГБ")
