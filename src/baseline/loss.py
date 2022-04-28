import torch


class LabelSmoothingLoss(torch.nn.Module):
    """formula
    loss= {
        (1-smoothing) * logP(x), if (x==y)
        (smoothing) / (num_classes-1) * logP(x), if (x!=y)
    }
    Args:
        torch (_type_): _description_
    """
    def __init__(self, smoothing:float=0.1, reduction:str='mean', ignore_index:int=-100):
        assert reduction in ('mean', 'sum', 'none')
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self._reduction = reduction
        self._ignore_index = ignore_index

    def forward(self, pred:torch.Tensor, target:torch.Tensor):
        num_classes = pred.size()[-1]
        pred = pred.log_softmax(dim=-1)
        
        pred = pred[target != self._ignore_index]
        target = target[target != self._ignore_index]
        
        new_target = torch.zeros_like(pred)
        new_target.fill_(value=self.smoothing / (num_classes - 1))
        new_target.scatter_(dim=1, index=target.data.unsqueeze(1), value=self.confidence)
        loss = -new_target * pred
        if self._reduction == 'mean':
            return torch.mean(torch.sum(loss, -1))
        elif self._reduction == 'sum':
            return torch.sum(loss, -1)
        elif self._reduction == 'none':
            return loss
    
    