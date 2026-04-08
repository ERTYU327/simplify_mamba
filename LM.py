import torch
from torch import nn

from Baiju_model import BaijuFlex


class LM(nn.Module):
    def __init__(self, d_model,d_state,first_block,other_block):
        super(LM, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.BaijuFlex1 = BaijuFlex(d_model,d_state,first_block,other_block)
        self.BaijuFlex2 = BaijuFlex(d_model,d_state,first_block,other_block)
        self.BaijuFlex3 = BaijuFlex(d_model, d_state, first_block,other_block)
        self.BaijuFlex4 = BaijuFlex(d_model, d_state, first_block,other_block)
        self.BaijuFlex5 = BaijuFlex(d_model, d_state, first_block,other_block)
        self.BaijuFlex6 = BaijuFlex(d_model, d_state, first_block,other_block)
        self.res=nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.SiLU(),
            nn.Linear(d_model,d_model),
        )
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(p=0.05)
    def forward(self,x):
        x1 = self.BaijuFlex1(x)
        res = self.res(x)
        x1 = res + x1
        x2 = self.BaijuFlex2(x1)
        res = self.res(x1)
        x2 = res + x2
        x3 = self.BaijuFlex3(x2)
        res = self.res(x2)
        x3 = res + x3
        x4 = self.BaijuFlex4(x3)
        res = self.res(x3)
        x4 = res + x4
        x5 = self.BaijuFlex5(x4)
        res = self.res(x4)
        x5 = res + x5
        x6 = self.BaijuFlex6(x5)
        x6 = self.dropout(x6)
        return x6

    def step_generate(self, x, state=None):
        B,L,_ = x.shape
        if state is None:
            state = []
            for i in range(6):
                h = torch.zeros(B,L,self.d_state).to(x.device)
                state.append(h)
        x1,h1 = self.BaijuFlex1.step_generate(x, state[0])
        res = self.res(x)
        x1 = res + x1
        x2,h2 = self.BaijuFlex2.step_generate(x1, state[1])
        res = self.res(x1)
        x2 = res + x2
        x3,h3 = self.BaijuFlex3.step_generate(x2, state[2])
        res = self.res(x2)
        x3 = res + x3
        x4,h4 = self.BaijuFlex4.step_generate(x3, state[3])
        res = self.res(x3)
        x4 = res + x4
        x5,h5 = self.BaijuFlex5.step_generate(x4, state[4])
        res = self.res(x4)
        x5 = res + x5
        x6,h6 = self.BaijuFlex5.step_generate(x5, state[5])
        return x6,[h1,h2,h3,h4,h5,h6]
