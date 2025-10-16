from transformers import AutoImageProcessor, AutoBackbone
model = AutoBackbone.from_pretrained("facebook/dinov2-base", out_features=["stage2", "stage5", "stage8", "stage11"])
# print weights
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
        
print(model.config.patch_size)