# focal_loss_pytorch

## Example

```py
criterion = FocalLoss()

x = torch.rand(1, 4)
y = torch.ones(1, dtype=torch.long)
# x: tensor([[0.8887, 0.3262, 0.1957, 0.8192]])
# y: tensor([1]))

criterion(x, y)
# tensor(1.6621)
```

## Reference

- https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html
- https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py
- https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/4
- https://github.com/Kitsunetic/focal_loss_pytorch
