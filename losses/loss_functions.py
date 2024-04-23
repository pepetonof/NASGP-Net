import torch
import torch.nn as nn
import torch.nn.functional as F

def one_hot(labels, num_classes, device, dtype):
    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

class DiceLoss(nn.Module):
    def __init__(self, 
                 average:str="micro", 
                 include_background:bool=False,
                 softmax:bool=True,
                 eps:float=1e-8) -> None:
        super().__init__()
        self.average = average
        self.eps = eps
        self.include_background=include_background
        self.softmax=softmax
    
    def forward(self, _input:torch.Tensor, _target:torch.Tensor):
        # if not len(_input.shape) == 4 or not len(_input.shape) == 5:
        #     raise ValueError(f"Invalid input shape, we expect BxNxHxW or BxNxHxWxD  Got: {_input.shape}")

        # if not _input.shape[-2:] == _target.shape[-2:]:
        #     raise ValueError(f"input and target shapes must be the same. Got: {_input.shape} and {_target.shape}")
        num_classes = _input.shape[1]
        dims = torch.arange(2, len(_input.shape)).tolist()
        
        #'macro':Calculate the metric for each class separately, and average the metrics across classes
        if self.average == "micro": #Calculate the metric globally, across all samples and classes.
            dims = (1, *dims)
        
        if not _input.device == _target.device:
            raise ValueError(f"input and target must be in the same device. Got: {_input.device} and {_target.device}")
        
        if self.softmax:
            _input = F.softmax(_input, dim=1)
        
        _target = one_hot(_target, num_classes=num_classes,
                                 device=_input.device, dtype=_input.dtype)
        
        #Ignore background?
        if not self.include_background:
            _input = _input[:, 1:]
            _target = _target[:, 1:]
        
        if _input.shape!=_target.shape:
            raise ValueError(f"input and target shapes must be the same. Got: {_input.shape} and {_target.shape}")
        
        # print(_input.device, _target.device, _input.shape, _target.shape, _input.dtype, _target.dtype)
        # print(_)
        # print(torch.amax(_target,(1,2,3,4)), torch.amin(_target,(1,2,3,4)))
        
        # compute the actual dice score
        intersection = torch.sum(_input * _target, dims)
        # print(intersection.shape)
        cardinality = torch.sum(_input + _target, dims)
        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        dice_loss = -dice_score + 1.0
        
        # reduce the loss across samples (and classes in case of `macro` averaging)
        dice_loss = torch.mean(dice_loss)
        
        del _input
        del _target
        
        return  dice_loss