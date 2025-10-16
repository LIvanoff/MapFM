# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train MapFM with 1 GPU with gradient accumulation
```
./tools/dist_train.sh ./projects/configs/mapfm/mapfm_nusc_dino2s_24ep.py 1
```

Eval MapFM with 1 GPU
```
./tools/dist_test_map.sh ./projects/configs/mapfm/mapfm_nusc_dino2s_24ep.py ./path/to/ckpts.pth 1
```




# Visualization 

we provide tools for visualization and benchmark under `path/to/MapFM/tools/maptrv2`