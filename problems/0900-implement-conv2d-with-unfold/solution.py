import torch
import torch.nn.functional as F

def conv2d_via_unfold(x, W, b, stride=1, padding=0):
    # TODO: implement conv2d using F.unfold + matmul
    n,_,h,w=x.shape
    flatten_w=torch.flatten(W,start_dim=1)
    cout,_,kh,kw=W.shape
    col=F.unfold(input=x,kernel_size=(kh,kw),padding=padding,stride=stride)
    ho = (h + 2*padding - kh) // stride + 1
    wo= (w + 2*padding - kw) // stride + 1
    out=torch.matmul(flatten_w,col)
    return out.view(n,cout,ho,wo)+b.view(1,cout,1,1)
