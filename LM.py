from torch import nn

from BaijuFlex import BaijuFlex


class LM(nn.Module):
    def __init__(self, d_model,d_state,num_blocks):
        super(LM, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.BaijuFlex1 = BaijuFlex(d_model,d_state,num_blocks)
        self.res=nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.SiLU(),
            nn.Linear(d_model,d_model),
        )
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(p=0.00)
    def forward(self,x,seq_len):
        x1,loss1 = self.BaijuFlex1(x,seq_len)
        return x1 , loss1

    def step(self, x, seq_len):
        x1 = self.BaijuFlex1.step(x,seq_len)
        return x1