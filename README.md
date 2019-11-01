### Tips:
  1. **TORCH_HOME:** used to custom model dir when downloading pretrained models from model zoos;

### Datasets:
  1. **Split2TnV:** used to create train and validation set in the form of txt file;
  2. **ItemList:** taken the txt files create from Split2TnV and used to create pytorch datasets which then can be load with pytorch dataloader;


### Losses:
  1. **SVSoftmax:** Support Vector Guided Softmax Loss for Face Recognition: https://arxiv.org/pdf/1812.11317.pdf;
  2. **Arcface:** ArcFace: Additive Angular Margin Loss for Deep Face Recognition: https://arxiv.org/pdf/1801.07698.pdf;


### Schedulers:
  1. **CyclicalLR:** Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/pdf/1506.01186.pdf;
  2. **CosineLR:** SGDR: STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS: https://arxiv.org/pdf/1608.03983.pdf;


---
### CosineAnnealingWarmRestarts(1.3.0):
  1. **T_0:** cycle length;
  2. **T_mult:** cycle length multiplier;