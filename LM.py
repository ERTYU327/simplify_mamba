from torch import nn

from Baiju_model import BaijuFlex


class LM(nn.Module):
    def __init__(self, d_model,d_state,first_block,other_block):
        super(LM, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.BaijuFlex1 = BaijuFlex(d_model,d_state,first_block,other_block)
        self.BaijuFlex2 = BaijuFlex(d_model,d_state,first_block,other_block)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(p=0.05)
    def forward(self,x):
        x1 = self.BaijuFlex1(x)
        x1 = self.act(x) + x1
        x2 = self.BaijuFlex2(self.act(x1))
        x2 = self.act(x) + self.act(x1) + x2
        return x2

    def step(self, x, states=None):
        if states is None:
            states = [None] * 2
        x1, h1 = self.BaijuFlex1.step_generate(x, states[0])
        x1 = self.act(x) + x1
        x2, h2 = self.BaijuFlex2.step_generate(x1, states[1])
        x2 = self.act(x) + self.act(x1) + x2
        return x2, [h1, h2]