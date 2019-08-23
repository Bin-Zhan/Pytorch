import torch
from torch import nn



class Arcface(nn.Module):
    """
    An implementation of ArcFace: Additive Angular Margin Loss for Deep Face Recognition: https://arxiv.org/pdf/1801.07698.pdf

    Args:
        embedding_size (int): Feature dimension.
        calssnum (int): Number of total classes.
        m (float): Margin value, see the paper for details. Default: 0.5.
        s (float): The scale value, see the paper for details. Default: 64.
    """
    def __init__(self, embedding_size, classnum, m=0.5, s=64.):
        super(Arcface, self).__init__()
        # initial kernel
        self.kernel = nn.Parameter(torch.empty(classnum, embedding_size))
        nn.init.xavier_uniform_(self.kernel)

        self.classnum = classnum
        self.s = s
        self.m = m

    def forward(self, embbedings, label):

        if not self.training:
            self.m = 0.

        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)

        cos_theta = F.linear(F.normalize(embbedings), F.normalize(self.kernel))
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        cos_theta_m = torch.where(cos_theta > self.m, cos_theta_m, cos_theta)

        # one hot encoding label
        one_hot = torch.zeros_like(cos_theta).scatter_(1, label.view(-1, 1), 1)

        output = (one_hot * cos_theta_m) + (1.0 - one_hot) * cos_theta

        return output * self.s



class SVSoftmax(nn.Module):
    """
    An implementation of Support Vector Guided Softmax Loss for Face Recognition: https://arxiv.org/pdf/1812.11317.pdf

    Args:
        embedding_size (int): Feature dimension, e.g. 512.
        classnum (int): Number of total classes.
        s (float): The scale value. Default: 30.
        t (float): Indicator parameter, see the paper for detailed introduction. Default: 1.2.
        m: Margin value used in Arcface Loss. Default: 0.5.
    """
    def __init__(self, embedding_size, classnum, s=30., t=1.2, m=0.5):
        super(SVSoftmax, self).__init__()
        # initial kernel
        self.kernel = nn.Parameter(torch.empty(classnum, embedding_size))
        nn.init.xavier_uniform_(self.kernel)

        self.classnum = classnum
        self.m = m
        self.s = s
        self.t = t

    def forward(self, embbedings, label):

        if not self.training:
            self.m = 0.
            self.t = 1

        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)

        cos_theta = F.linear(F.normalize(embbedings), F.normalize(self.kernel))
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        cos_theta_t = self.t * cos_theta + self.t - 1

        # one hot encoding label
        one_hot = torch.zeros_like(cos_theta).scatter_(1, label.view(-1, 1), 1)
        # get predicted label from cos_theta
        p_label = cos_theta[np.arange(cos_theta.size(0)), label].view(-1, 1)

        cos_theta_m = torch.where(cos_theta > self.m, cos_theta_m, cos_theta)
        output = (one_hot * cos_theta_m) + (1.0 - one_hot) * torch.where(((cos_theta > 0) & (cos_theta <= p_label)), cos_theta, cos_theta_t)

        return output * self.s
