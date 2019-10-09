######################### custom torchvision model zoo dir #########################
import os
os.environ['TORCH_HOME'] = './' # pretrained model will be saved to dir: './checkpoints'; default dir is '~/.cache/torch/checkpoints'

import torchvision

vgg = torchvision.models.vgg19(pretrained=True, progress=True)
print(vgg)
######################### custom torchvision model zoo dir #########################