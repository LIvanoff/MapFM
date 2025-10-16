import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import List, Optional, Dict, Any
from mmdet.models.builder import BACKBONES
from mmcv.runner.base_module import BaseModule

from dinov2_hf import Dinov2Backbone
# projects/mmdet3d_plugin/models/backbones/dinov2_hf.py

# Предполагаем, что Dinov2Backbone импортирован или определен где-то
# Пример импорта из transformers:
# from transformers import Dinov2Backbone

# --- Новый модуль-обертка для ОДНОГО линейного слоя ---
class _LoRA_Linear(nn.Module):
    """
    Обертка для одного nn.Linear слоя для применения LoRA.
    Добавляет выход LoRA-пути (W_b * W_a * x) к выходу оригинального слоя.
    """
    def __init__(
        self,
        original_linear: nn.Linear,
        linear_a: nn.Linear,
        linear_b: nn.Linear,
        layer_norm: Optional[nn.Module] = None, # Опциональная нормализация перед LoRA
    ):
        super().__init__()
        self.original_linear = original_linear
        self.linear_a = linear_a
        self.linear_b = linear_b
        self.layer_norm = layer_norm if layer_norm is not None else nn.Identity()

        # Сохраняем размерности для информации
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = linear_a.out_features

    def forward(self, x: Tensor) -> Tensor:
        original_output = self.original_linear(x)
        
        # Применяем LayerNorm перед LoRA, если он есть
        lora_input = self.layer_norm(x)
        
        # Вычисляем LoRA "дельту"
        lora_output = self.linear_b(self.linear_a(lora_input))

        # Добавляем дельту к оригинальному выходу
        return original_output + lora_output

# --- Адаптированный класс LoRA для Hugging Face DINOv2 ---
class LoRA(BaseModule):
    def __init__(
        self,
        dinov2_model: Dinov2Backbone,
        r: int=32,                 # Ранг LoRA
        lora_layer_ids: Optional[List[int]] = None, # Индексы слоев для LoRA (None = все)
        use_layer_norm: bool = False, # Использовать ли LayerNorm перед LoRA
        lora_k: bool = True,          # Применять ли LoRA к Key проекции
        lora_v: bool = True,          # Применять ли LoRA к Value проекции (рекомендуется True)
        lora_q: bool = True,          # Применять ли LoRA к Query проекции (рекомендуется True)
    ):
        super(LoRA, self).__init__()
        
        assert r > 0, "LoRA rank 'r' must be positive."
                
        self.r = r
        self.lora_layer_ids = lora_layer_ids if lora_layer_ids is not None else list(range(len(dinov2_model.encoder.layer)))
        self.use_layer_norm = use_layer_norm
        self.lora_k = lora_k
        self.lora_v = lora_v
        self.lora_q = lora_q

        # Хранилища для LoRA слоев (A и B матрицы)
        self.w_As: nn.ModuleList = nn.ModuleList()
        self.w_Bs: nn.ModuleList = nn.ModuleList()
                    

        # "Хирургия": встраиваем LoRA слои
        for layer_idx, blk in enumerate(dinov2_model.encoder.layer):
            if layer_idx not in self.lora_layer_ids:
                continue

            # Доступ к слоям внимания Q, K, V
            # Путь: Dinov2Layer -> Dinov2Attention -> Dinov2SelfAttention -> [query, key, value]
            attn_module = blk.attention.attention
            original_query = attn_module.query
            original_key = attn_module.key
            original_value = attn_module.value

            dim = original_query.in_features

            # --- Обработка Query ---
            if self.lora_q:
                w_a_linear_q = nn.Linear(dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, dim, bias=False)
                ln_q = nn.LayerNorm(dim) if use_layer_norm else None
                
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                
                attn_module.query = _LoRA_Linear(
                    original_query, w_a_linear_q, w_b_linear_q, ln_q
                )
                print(f"Applied LoRA to Query in layer {layer_idx}")

            # --- Обработка Key ---
            if self.lora_k:
                w_a_linear_k = nn.Linear(dim, r, bias=False)
                w_b_linear_k = nn.Linear(r, dim, bias=False)
                ln_k = nn.LayerNorm(dim) if use_layer_norm else None

                self.w_As.append(w_a_linear_k)
                self.w_Bs.append(w_b_linear_k)

                attn_module.key = _LoRA_Linear(
                    original_key, w_a_linear_k, w_b_linear_k, ln_k
                )
                print(f"Applied LoRA to Key in layer {layer_idx}")

            # --- Обработка Value ---
            if self.lora_v:
                w_a_linear_v = nn.Linear(dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, dim, bias=False)
                ln_v = nn.LayerNorm(dim) if use_layer_norm else None

                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)

                attn_module.value = _LoRA_Linear(
                    original_value, w_a_linear_v, w_b_linear_v, ln_v
                )
                print(f"Applied LoRA to Value in layer {layer_idx}")

        self.reset_parameters() # Инициализация LoRA весов
        self.lora_dinov2 = dinov2_model # Сохраняем модифицированную модель

        # Размораживаем только параметры LoRA и LayerNorm (если используется)
        for param in self.w_As.parameters():
            param.requires_grad = True
        for param in self.w_Bs.parameters():
            param.requires_grad = True
            
        if self.use_layer_norm:
            for blk in self.lora_dinov2.encoder.layer:
                 if hasattr(blk.attention.attention.query, 'layer_norm') and blk.attention.attention.query.layer_norm is not None:
                     for param in blk.attention.attention.query.layer_norm.parameters():
                         param.requires_grad = True
                 if hasattr(blk.attention.attention.key, 'layer_norm') and blk.attention.attention.key.layer_norm is not None:
                     for param in blk.attention.attention.key.layer_norm.parameters():
                         param.requires_grad = True
                 if hasattr(blk.attention.attention.value, 'layer_norm') and blk.attention.attention.value.layer_norm is not None:
                     for param in blk.attention.attention.value.layer_norm.parameters():
                         param.requires_grad = True


    def save_lora_parameters(self, filename: str) -> None:
        """Сохраняет параметры ТОЛЬКО LoRA слоев (A и B) в файл .pt"""
        assert filename.endswith(".pt"), "Filename must end with .pt"

        lora_state_dict = {}
        for i, w_A in enumerate(self.w_As):
            lora_state_dict[f"w_a_{i:03d}"] = w_A.state_dict()
        for i, w_B in enumerate(self.w_Bs):
             lora_state_dict[f"w_b_{i:03d}"] = w_B.state_dict()
        
        # Опционально: сохраняем параметры LayerNorm, если они использовались и обучались
        if self.use_layer_norm:
            ln_state_dict = {}
            ln_idx = 0
            for layer_idx, blk in enumerate(self.lora_dinov2.encoder.layer):
                 if layer_idx not in self.lora_layer_ids:
                      continue
                 
                 def add_ln_state(module, prefix):
                      nonlocal ln_idx
                      if hasattr(module, 'layer_norm') and isinstance(module.layer_norm, nn.LayerNorm):
                           ln_state_dict[f"{prefix}_ln_{ln_idx:03d}"] = module.layer_norm.state_dict()
                           ln_idx += 1
                           
                 if self.lora_q: add_ln_state(blk.attention.attention.query, f"query_layer_{layer_idx}")
                 if self.lora_k: add_ln_state(blk.attention.attention.key,   f"key_layer_{layer_idx}")
                 if self.lora_v: add_ln_state(blk.attention.attention.value, f"value_layer_{layer_idx}")

            if ln_state_dict:
                 lora_state_dict['lora_layer_norms'] = ln_state_dict


        torch.save(lora_state_dict, filename)
        print(f"LoRA parameters saved to {filename}")

    def load_lora_parameters(self, filename: str) -> None:
        """Загружает параметры LoRA слоев (A и B) из файла .pt"""
        assert filename.endswith(".pt"), "Filename must end with .pt"

        state_dict = torch.load(filename, map_location='cpu') # Загружаем на CPU

        # Загрузка весов A и B
        for i, w_A in enumerate(self.w_As):
            key = f"w_a_{i:03d}"
            if key in state_dict:
                w_A.load_state_dict(state_dict[key])
            else:
                print(f"Warning: LoRA weight key {key} not found in checkpoint.")
        
        for i, w_B in enumerate(self.w_Bs):
            key = f"w_b_{i:03d}"
            if key in state_dict:
                w_B.load_state_dict(state_dict[key])
            else:
                print(f"Warning: LoRA weight key {key} not found in checkpoint.")

        # Опционально: загрузка весов LayerNorm
        if self.use_layer_norm and 'lora_layer_norms' in state_dict:
             ln_state_dict = state_dict['lora_layer_norms']
             ln_idx = 0
             for layer_idx, blk in enumerate(self.lora_dinov2.encoder.layer):
                 if layer_idx not in self.lora_layer_ids:
                      continue
                 
                 def load_ln_state(module, prefix):
                     nonlocal ln_idx
                     if hasattr(module, 'layer_norm') and isinstance(module.layer_norm, nn.LayerNorm):
                         ln_key = f"{prefix}_ln_{ln_idx:03d}"
                         if ln_key in ln_state_dict:
                             module.layer_norm.load_state_dict(ln_state_dict[ln_key])
                         else:
                             print(f"Warning: LoRA LayerNorm key {ln_key} not found in checkpoint.")
                         ln_idx += 1

                 if self.lora_q: load_ln_state(blk.attention.attention.query, f"query_layer_{layer_idx}")
                 if self.lora_k: load_ln_state(blk.attention.attention.key,   f"key_layer_{layer_idx}")
                 if self.lora_v: load_ln_state(blk.attention.attention.value, f"value_layer_{layer_idx}")

        print(f"LoRA parameters loaded from {filename}")

    def reset_parameters(self) -> None:
        """Инициализация LoRA весов."""
        for w_A in self.w_As:
            # Инициализация Kaiming для матрицы A
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            # Инициализация нулями для матрицы B
            nn.init.zeros_(w_B.weight)
        
        # Опционально: инициализация LayerNorm (обычно стандартная инициализация уже хороша)
        if self.use_layer_norm:
            for blk in self.lora_dinov2.encoder.layer:
                 # Стандартная инициализация LN: weight=1, bias=0
                 if hasattr(blk.attention.attention.query, 'layer_norm') and isinstance(blk.attention.attention.query.layer_norm, nn.LayerNorm):
                      nn.init.ones_(blk.attention.attention.query.layer_norm.weight)
                      nn.init.zeros_(blk.attention.attention.query.layer_norm.bias)
                 if hasattr(blk.attention.attention.key, 'layer_norm') and isinstance(blk.attention.attention.key.layer_norm, nn.LayerNorm):
                      nn.init.ones_(blk.attention.attention.key.layer_norm.weight)
                      nn.init.zeros_(blk.attention.attention.key.layer_norm.bias)
                 if hasattr(blk.attention.attention.value, 'layer_norm') and isinstance(blk.attention.attention.value.layer_norm, nn.LayerNorm):
                      nn.init.ones_(blk.attention.attention.value.layer_norm.weight)
                      nn.init.zeros_(blk.attention.attention.value.layer_norm.bias)


    # Убираем forward метод из этого класса, так как он модифицирует
    # модель на месте. Пользователь должен использовать self.lora_dinov2
    # def forward(self, x: Tensor) -> Tensor:
    #     return self.lora_dinov2(x)
    
    
# @BACKBONES.register_module()
# class DINOv2_LoRA(BaseModule):
#     def __init__(self, config,
#                  out_features=["stage12"],
#                  ignore_mismatched_sizes=True,
#                  output_hidden_states=True,
#                  frozen_stages=13,
#                  r=32, 
#                  lora_k=False,
#                  lora_q=True,
#                  lora_v=True,
#                  use_layer_norm=False,
#                  lora_layer_ids=None 
#                  ):
#         super(DINOv2_LoRA, self).__init__()
#         self.config = config
#         self.out_features = out_features
#         self.ignore_mismatched_sizes = ignore_mismatched_sizes
#         self.output_hidden_states = output_hidden_states
#         self.frozen_stages = frozen_stages
#         self.lora_rank = r
#         self.lora_k = lora_k
#         self.lora_q = lora_q
#         self.lora_v = lora_v
#         self.use_layer_norm = use_layer_norm
#         self.lora_layer_ids = lora_layer_ids
        
#         self.dinov2 = Dinov2Backbone.from_pretrained(self.config,
#                                                     out_features = self.out_features,
#                                                     ignore_mismatched_sizes = self.ignore_mismatched_sizes
#                                                     )
#         self._freeze_stages()
        
#         lora_adapter = LoRA(self.dinov2, 
#                             r=self.lora_rank, 
#                             lora_k=self.lora_k, # Не применять к Key
#                             lora_q=self.lora_q,  # Применять к Query
#                             lora_v=self.lora_v,  # Применять к Value
#                             use_layer_norm=False,
#                             lora_layer_ids=self.lora_layer_ids 
#                         )
#         self.model = lora_adapter.lora_dinov2
        
#         print("\nTrainable parameters:")
#         total_params = 0
#         trainable_params = 0
#         for name, param in self.model.named_parameters():
#             total_params += param.numel()
#             if param.requires_grad:
#                 print(f"- {name} ({param.numel()})")
#                 trainable_params += param.numel()
#         print(f"\nTotal parameters: {total_params}")
#         print(f"Trainable parameters (LoRA): {trainable_params}")
#         print(f"Trainable ratio: {trainable_params / total_params * 100:.4f}%")
        
        
        
#     def forward(self, x):
#         outputs = self.model(x)
#         feature_maps = outputs.feature_maps
#         # import ipdb; ipdb.set_trace()
#         return feature_maps
    
#     def _freeze_stages(self):
#         if self.frozen_stages >= 0:
#             self.dinov2.embeddings.eval()
#             for param in self.dinov2.embeddings.parameters():
#                 param.requires_grad = False

#         if self.frozen_stages >= 1:
#             for i in range(0, self.frozen_stages - 1):
#                 m = self.dinov2.encoder.layer
#                 m.eval()
#                 for param in m.parameters():
#                     param.requires_grad = False
#             if not self.use_layer_norm:
#                 for param in self.dinov2.layernorm.parameters():
#                     param.requires_grad = False
                


# --- Пример использования ---
# if __name__ == '__main__':
#     from transformers import Dinov2Backbone, Dinov2Config

#     # 1. Загрузка предобученной модели DINOv2
#     # Используем конфигурацию dinov2-base (num_layers=12, hidden_size=768)
#     # Убедитесь, что конфигурация соответствует вашей распечатке модели
#     config = 'projects/mmdet3d_plugin/models/backbones/dinov2-base'
#     model = Dinov2Backbone.from_pretrained('projects/mmdet3d_plugin/models/backbones/dinov2-base', 
#                                        out_features=["stage12"],
#                                        ignore_mismatched_sizes=True
#                                        )
#     # original_model = Dinov2Backbone(...) # Ваша модель из распечатки
    
#     # Распечатка структуры (для сверки)
#     # print("Original Model Structure:")
#     # print(original_model)

#     # 2. Создание LoRA-адаптированной модели
#     lora_rank = 32 # Ранг LoRA (типичные значения: 4, 8, 16, 32)
    
#     # Применяем LoRA ко всем слоям, к Q, V (но не K), без LayerNorm
#     lora_adapter = LoRA(
#         model, 
#         r=lora_rank, 
#         lora_k=False, # Не применять к Key
#         lora_q=True,  # Применять к Query
#         lora_v=True,  # Применять к Value
#         use_layer_norm=False 
#     )
    
#     # Получаем модифицированную модель
#     lora_model = lora_adapter.lora_dinov2

#     # 3. Проверка trainable параметров
#     print("\nTrainable parameters:")
#     total_params = 0
#     trainable_params = 0
#     for name, param in lora_model.named_parameters():
#         total_params += param.numel()
#         if param.requires_grad:
#             print(f"- {name} ({param.numel()})")
#             trainable_params += param.numel()
            
#     print(f"\nTotal parameters: {total_params}")
#     print(f"Trainable parameters (LoRA): {trainable_params}")
#     print(f"Trainable ratio: {trainable_params / total_params * 100:.4f}%")

#     # 4. Пример использования модели (прогон случайного тензора)
#     dummy_input = torch.randn(1, 3, 448, 784) # Пример батча из 1 картинки 224x224
    
#     try:
#         # Модель Dinov2Backbone обычно возвращает BaseModelOutputWithPooling
#         # или просто тензор последнего скрытого состояния
#         with torch.no_grad(): # Не считаем градиенты для простого прогона
#              outputs = lora_model(dummy_input)
#         feature_maps = outputs.feature_maps

#         print("Output shape (last_hidden_state):", feature_maps[-1].shape) 
#         # Ожидаемая форма: [batch_size, num_patches + 1(cls_token), hidden_size]
#         # Для 224x224 и patch_size=14: num_patches = (224/14)^2 = 16^2 = 256
#         # Ожидаемый shape: [1, 257, 768]

#     except Exception as e:
#         print(f"\nError during model forward pass: {e}")
#         print("Check the input dimensions and model configuration.")

#     # 5. Сохранение и загрузка LoRA параметров
#     lora_weights_file = "dinov2_lora_r8.pt"
#     lora_adapter.save_lora_parameters(lora_weights_file)
    
#     lora_adapter_load = LoRA(
#         model, 
#         r=lora_rank, 
#         lora_k=False, # Важно использовать те же настройки!
#         lora_q=True,
#         lora_v=True,
#         use_layer_norm=False
#     )
    
#     # Загружаем веса
#     lora_adapter_load.load_lora_parameters(lora_weights_file)
#     loaded_lora_model = lora_adapter_load.lora_dinov2
#     print(f"\nSuccessfully loaded LoRA parameters into a new model instance.")
    
#     dinov2_lora = DINOv2_LoRA(config=config)