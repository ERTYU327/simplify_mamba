from pickletools import uint2

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAMBA(nn.Module):
    def __init__(self, d_model=64, d_state=16):
        super(MAMBA, self).__init__()
        self.d_model = d_model
        self.d_state = d_state

        # 输入投影层
        self.in_proj = nn.Linear(d_model, d_state, bias=True)

        # SSM参数投影层
        self.A1_proj = nn.Linear(d_model, d_state // 4, bias=True)
        self.A2_proj = nn.Linear(d_model, d_state // 8, bias=True)
        self.A3_proj = nn.Linear(d_model, d_state // 8, bias=True)
        self.A4_proj = nn.Linear(d_model, d_state // 8, bias=True)
        self.A5_proj = nn.Linear(d_model, d_state // 8, bias=True)
        self.A6_proj = nn.Linear(d_model, d_state // 8, bias=True)
        self.A7_proj = nn.Linear(d_model, d_state // 8, bias=True)
        self.u1_proj = nn.Linear(d_model, d_state // 4, bias=True)
        self.u2_proj = nn.Linear(d_model, d_state // 8, bias=True)
        self.u3_proj = nn.Linear(d_model, d_state // 8, bias=True)
        self.u4_proj = nn.Linear(d_model, d_state // 8, bias=True)
        self.u5_proj = nn.Linear(d_model, d_state // 8, bias=True)
        self.u6_proj = nn.Linear(d_model, d_state // 8, bias=True)
        self.u7_proj = nn.Linear(d_model, d_state // 8, bias=True)
        self.B_proj = nn.Linear(d_model, d_state, bias=True)
        self.C_proj = nn.Linear(d_model, d_state, bias=True)
        self.dt_proj = nn.Linear(d_model, d_state, bias=True)

        # 输出投影层
        self.out_proj = nn.Linear(d_state, d_model, bias=True)
        self.A_fixed = nn.Parameter(torch.randn(d_state) * 0.1)  # 可训练，但不依赖输入

    def step(self, u, h=None):
        batch_size = u.shape[0]

        # 输入投影
        x = self.in_proj(u)  # (B, D)

        # 计算SSM参数
        A1 = self.A1_proj(u)  # (B, D_state)
        A2 = self.A2_proj(u)
        A3 = self.A3_proj(u)
        A4 = self.A4_proj(u)
        A5 = self.A5_proj(u)
        A6 = self.A6_proj(u)
        A7 = self.A7_proj(u)
        u1 = self.u1_proj(u)
        u2 = self.u2_proj(u)
        u3 = self.u3_proj(u)
        u4 = self.u4_proj(u)
        u5 = self.u5_proj(u)
        u6 = self.u6_proj(u)
        u7 = self.u7_proj(u)
        point1 = torch.mul(A1, u1).sum(dim=1, keepdim=True)
        point2 = torch.mul(A2, u2).sum(dim=1, keepdim=True)
        point3 = torch.mul(A3, u3).sum(dim=1, keepdim=True)
        point4 = torch.mul(A4, u4).sum(dim=1, keepdim=True)
        point5 = torch.mul(A5, u5).sum(dim=1, keepdim=True)
        point6 = torch.mul(A6, u6).sum(dim=1, keepdim=True)
        point7 = torch.mul(A7, u7).sum(dim=1, keepdim=True)
        scores = torch.cat([point1, point2, point3, point4, point5, point6, point7], dim=1)  # (B, 3)
        weights = F.softmax(scores, dim=1)  # (B, 3)
        A1_weighted = weights[:, 0:1] * A1  # (B, d_state//2)
        A2_weighted = weights[:, 1:2] * A2  # (B, d_state//4)
        A3_weighted = weights[:, 2:3] * A3  # (B, d_state//4)
        A4_weighted = weights[:, 3:4] * A4
        A5_weighted = weights[:, 4:5] * A5
        A6_weighted = weights[:, 5:6] * A6
        A7_weighted = weights[:, 6:7] * A7
        A = torch.cat([A1_weighted, A2_weighted, A3_weighted, A4_weighted, A5_weighted, A6_weighted, A7_weighted],
                      dim=1)  # (B, d_state)
        B = self.B_proj(u)  # (B, D_state)
        C = self.C_proj(u)  # (B, D_state)
        dt = self.dt_proj(u)  # (B, D_state)
        # A = self.A_fixed.unsqueeze(0).expand(batch_size, -1)  # (B, D_state)
        # 离散化参数
        A = -torch.exp(A)  # 确保A是负的
        dt = F.softplus(dt)  # 确保dt是正的
        dA = torch.exp(dt * A)
        b = dA.size(0)
        c = dA.size(1)
        I = torch.ones(b, c, device=u.device)
        dB = (torch.exp(dA - I) / A) * B
        # dB = dt * B

        # 初始化状态
        if h is None:
            h = torch.zeros(batch_size, self.d_state, device=u.device)

        h_new = dA * h + dB * x.unsqueeze(1)  # (B, D_state)

        # 输出计算
        y = torch.sum(C * h_new, dim=1, keepdim=False)   # (B, 1)
        y = y.expand(-1, self.d_state)  # (B, D)
        y = self.out_proj(y)

        return y, h_new

    def forward(self, u):
        batch_size, seq_len, _ = u.shape
        # 输入投影
        x = self.in_proj(u)  # (B, L, D)

        outputs = []
        h = torch.zeros(batch_size, self.d_state, device=u.device)
        # A = self.A_fixed.unsqueeze(0).expand(batch_size, -1)  # (B, D_state)
        # A = -torch.exp(A)  # (B, D_state)
        for t in range(seq_len):
            x_t = x[:, t, :]  # (B, D)
            u_t = u[:, t, :]
            # 计算SSM参数
            A1 = self.A1_proj(u_t)  # (B, D_state)
            A2 = self.A2_proj(u_t)
            A3 = self.A3_proj(u_t)
            A4 = self.A4_proj(u_t)
            A5 = self.A5_proj(u_t)
            A6 = self.A6_proj(u_t)
            A7 = self.A7_proj(u_t)
            u1 = self.u1_proj(u_t)
            u2 = self.u2_proj(u_t)
            u3 = self.u3_proj(u_t)
            u4 = self.u4_proj(u_t)
            u5 = self.u5_proj(u_t)
            u6 = self.u6_proj(u_t)
            u7 = self.u7_proj(u_t)
            point1 = torch.mul(A1, u1).sum(dim=1, keepdim=True)
            point2 = torch.mul(A2, u2).sum(dim=1, keepdim=True)
            point3 = torch.mul(A3, u3).sum(dim=1, keepdim=True)
            point4 = torch.mul(A4, u4).sum(dim=1, keepdim=True)
            point5 = torch.mul(A5, u5).sum(dim=1, keepdim=True)
            point6 = torch.mul(A6, u6).sum(dim=1, keepdim=True)
            point7 = torch.mul(A7, u7).sum(dim=1, keepdim=True)
            scores = torch.cat([point1, point2, point3, point4, point5, point6, point7], dim=1)  # (B, 3)
            weights = F.softmax(scores, dim=1)  # (B, 3)
            A1_weighted = weights[:, 0:1] * A1  # (B, d_state//2)
            A2_weighted = weights[:, 1:2] * A2  # (B, d_state//4)
            A3_weighted = weights[:, 2:3] * A3  # (B, d_state//4)
            A4_weighted = weights[:, 3:4] * A4
            A5_weighted = weights[:, 4:5] * A5
            A6_weighted = weights[:, 5:6] * A6
            A7_weighted = weights[:, 6:7] * A7
            A = torch.cat([A1_weighted, A2_weighted, A3_weighted, A4_weighted, A5_weighted, A6_weighted, A7_weighted],
                          dim=1)  # (B, d_state)
            B = self.B_proj(u_t)  # (B, D_state)
            C = self.C_proj(u_t)  # (B, D_state)
            dt = self.dt_proj(u_t)  # (B, D_state)
            # 离散化参数
            A = -torch.exp(A)
            dt = F.softplus(dt)
            dA = torch.exp(dt * A)  # Ã = exp(ΔA)
            b = dA.size(0)
            c = dA.size(1)
            I = torch.ones(b, c, device=u.device)
            # dB = dt * B             # B̃ ≈ ΔB
            dB = (torch.exp(dA - I) / A) * B
            #h = dA * h + dB * x_t.unsqueeze(1)
            h = dA * h + dB * x_t
            # 输出计算
            #y_t = torch.sum(C * h, dim=1, keepdim=False)   # (B, 1)
            y_t = C * h
            #y_t = y_t.expand(-1, self.d_state)  # (B, D)
            y_t = self.out_proj(y_t)
            outputs.append(y_t)

        # 堆叠输出
        output = torch.stack(outputs, dim=1)  # (B, L, D)

        return output