from torch import nn

from Baiju_origional import BaijuFlex


class LM_factor(nn.Module):
    def __init__(self, d_model,d_state,first_block,other_block,order,dx=4):
        super(LM_factor, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.BaijuFlex = BaijuFlex(d_model, d_state, first_block, other_block, order, dx=dx)
        self.layer_norm = nn.LayerNorm(d_model)
    def forward(self,x,init):
        x1,h1 = self.BaijuFlex.forward(x, init)
        return x1, h1
    def step_generate(self, x, h):
        x1,h1 = self.BaijuFlex.step(x, h=h)
        return x1, h1
class LM_last(nn.Module):
    def __init__(self, d_model,d_state,first_block,other_block,order,dx=4):
        super(LM_last, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.BaijuFlex = BaijuFlex(d_model, d_state, first_block, other_block, order, dx=dx)
    def forward(self,x,init):
        x1,h1 = self.BaijuFlex.forward(x, init)
        return x1, h1
    def step_generate(self, x, h):
        x1,h1 = self.BaijuFlex.step(x, h=h)
        return x1, h1
class LM(nn.Module):
    def __init__(self, d_model,d_state,first_block,other_block,cengshu,order,dx=4):
        super(LM, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.cengshu = cengshu
        self.dropout = nn.Dropout(p=0.00)
        self.LM_lastone = LM_last(d_model,d_state,first_block,other_block,order,dx=dx)
        self.LM_total = nn.ModuleList()
        for i in range(cengshu):
            self.LM_total.append(LM_factor(d_model,d_state,first_block,other_block,order,dx=dx))
    def forward(self,x,init):
        if self.cengshu == 1:
           x,h_final = self.LM_lastone(x,init)
        else:
           h_next = 0
           for i in range(len(self.LM_total)):
               xi, hi = self.LM_total[i](x, init)
               x = xi
               init = hi
               h_next = init
           x, h_final = self.LM_lastone(x, h_next)
        x = self.dropout(x)
        return x,h_final
    def step_generate(self, x, h):
        if self.cengshu == 1:
           x,h_final = self.LM_lastone.step_generate(x,h)
        else:
           h_next = 0
           for i in range(len(self.LM_total)):
               xi, hi = self.LM_total[i].step_generate(x, h)
               x = xi
               h = hi
               h_next = h
           x, h_final = self.LM_lastone(x, h_next)
        return x, h_final
