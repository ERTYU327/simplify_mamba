import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,groups):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding,groups=groups)
        self.kernel_size = kernel_size

    def forward(self, x):
        padding = self.kernel_size - 1
        x = F.pad(x, (padding, 0), mode='constant', value=0)
        return self.conv(x)
def associative_scan(elems, combine):
    L = elems.shape[0]
    if L == 1:
        return elems
    left = elems[:L//2]
    right = elems[L//2:]
    left_scan = associative_scan(left, combine)
    last_left = left_scan[-1]
    right_combined = torch.stack([combine(last_left, x) for x in right], dim=0)
    return torch.cat([left_scan, right_combined], dim=0)


class BaijuFlex(nn.Module):
    def __init__(self, d_model=64, d_state=16, first_block=8, other_block=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        branch_dims = []
        if first_block == 1 and other_block == 0:
            first_dim = d_state
            other_dim = 0
            branch_dims.append(d_state)
        elif first_block == 1 and other_block == 1:
            first_dim = d_state // 2
            other_dim = d_state // 2
            branch_dims.append(d_state // 2)
            branch_dims.append(d_state // 2)
        else:
            first_dim = d_state // first_block
            other_dim = d_state // other_block
            branch_dims.append(first_dim)
            for i in range(other_block - 2):
                branch_dims.append(other_dim)
        self.branch_dims = branch_dims
        self.n_branches = len(branch_dims)
        self.in_proj = nn.Linear(d_model, d_state, bias=True)
        self.layer_norm = nn.LayerNorm(d_state)
        self.act = nn.Sigmoid()
        self.A_projs = nn.ModuleList()
        self.u_projs = nn.ModuleList()
        self.k_projs = nn.ModuleList()
        for dim in branch_dims:
            self.A_projs.append(nn.Sequential(
                nn.Linear(d_state, dim, bias=True),
                nn.Sigmoid(),
                nn.Linear(dim, dim, bias=True),
            ))
            self.u_projs.append(nn.Sequential(
                nn.Linear(d_model, dim, bias=True),
                nn.Sigmoid(),
                nn.Linear(dim, dim, bias=True),
            ))
            self.k_projs.append(nn.Sequential(
                nn.Linear(d_model, dim, bias=True),
                nn.Sigmoid(),
                nn.Linear(dim, dim, bias=True),
            ))
        self.P_chol = nn.ParameterList()
        for dim in branch_dims:
            self.P_chol.append(nn.Parameter(torch.eye(dim)))
        # 因果卷积
        self.causalconv1d = CausalConv1d(
            in_channels=d_model, out_channels=d_state,
            kernel_size=2, padding=0, groups=2
        )

        # B, C, dt 投影
        self.B_proj = nn.Sequential(
            nn.Linear(d_model, d_state, bias=True),
            nn.Sigmoid(),
            nn.Linear(d_state, d_state, bias=True),
        )
        self.C1_proj = nn.Sequential(
            nn.Linear(d_model, d_state, bias=True),
            nn.Sigmoid(),
            nn.Linear(d_state, d_state, bias=True),
            nn.Sigmoid(),
        )
        self.dt_proj = nn.Sequential(
            nn.Linear(d_model, d_state, bias=True),
            nn.Sigmoid(),
            nn.Linear(d_state, d_state, bias=True),
            nn.Sigmoid(),
        )

        # 输出投影
        self.out_proj = nn.Linear(d_state, d_model, bias=True)

        self.K_proj = nn.Sequential(
            nn.Linear(self.n_branches, d_state, bias=True),
            nn.Sigmoid(),
            nn.Linear(d_state, d_state, bias=True),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(p=0.1)  # 仅在输出时使用

    def _get_branch_params(self, h, u):

        A_list = []
        u_list = []
        k_list = []
        scale_list = []
        for i, dim in enumerate(self.branch_dims):
            A_i = self.A_projs[i](h)      # (B, L, dim)
            u_i = self.u_projs[i](u)      # (B, L, dim)
            k_i = self.k_projs[i](u)      # (B, L, dim)
            scale_i = dim ** 0.5
            A_list.append(A_i)
            u_list.append(u_i)
            k_list.append(k_i)
            scale_list.append(scale_i)
        return A_list, u_list, k_list, scale_list

    def _compute_scores_and_weights(self, A_list, u_list, k_list, scale_list):
        B, L, D = A_list[0].shape
        scores_dot_list = []
        scores_euc_list = []
        scores_liman_list    = []
        for i in range(self.n_branches):
            dot = (A_list[i] * u_list[i] / scale_list[i] * k_list[i]).sum(dim=2, keepdim=True)
            scores_dot_list.append(dot)
            diff = A_list[i] - u_list[i]
            dist = torch.norm(diff, p=2, dim=2, keepdim=True)
            euc = -dist / (scale_list[i] + 1e-8)
            scores_euc_list.append(euc)
            H = torch.tril(self.P_chol[i])
            M_metrix = H @ H.T
            liman = ((diff @ M_metrix) * diff).sum(dim=2, keepdim=True)
            scores_liman_list.append(liman)
        scores_dot = torch.cat(scores_dot_list, dim=2)
        scores_euc = torch.cat(scores_euc_list, dim=2)
        scores_liman = torch.cat(scores_liman_list, dim=2)
        causal_mask = torch.tril(torch.ones((1, L, 1), device=scores_dot.device, dtype=scores_dot.dtype))
        scores_dot = scores_dot.masked_fill(causal_mask == 0, float('-inf'))
        scores_euc = scores_euc.masked_fill(causal_mask == 0, float('-inf'))
        scores_liman = scores_liman.masked_fill(causal_mask == 0, float('-inf'))
        w_dot = F.softmax(scores_dot, dim=2)   # 点积权重
        w_euc = F.softmax(scores_euc, dim=2)   # 欧氏权重
        w_liman = F.softmax(scores_liman, dim=2)
        return scores_dot, scores_euc, scores_liman, w_dot, w_euc, w_liman
    def _compute_scores_and_weights_generate(self, A_list, u_list, k_list, scale_list):
        B, L, D = A_list[0].shape
        scores_dot_list = []
        scores_euc_list = []
        scores_liman_list    = []
        for i in range(self.n_branches):
            dot = (A_list[i] * u_list[i] / scale_list[i] * k_list[i]).sum(dim=2, keepdim=True)
            scores_dot_list.append(dot)
            diff = A_list[i] - u_list[i]
            dist = torch.norm(diff, p=2, dim=2, keepdim=True)
            euc = -dist / (scale_list[i] + 1e-8)
            scores_euc_list.append(euc)
            H = torch.tril(self.P_chol[i])
            M_metrix = H @ H.T
            liman = ((diff @ M_metrix) * diff).sum(dim=2, keepdim=True)
            scores_liman_list.append(liman)
        scores_dot = torch.cat(scores_dot_list, dim=2)
        scores_euc = torch.cat(scores_euc_list, dim=2)
        scores_liman = torch.cat(scores_liman_list, dim=2)
        causal_mask = torch.tril(torch.ones((1, L, 1), device=scores_dot.device, dtype=scores_dot.dtype))
        scores_dot = scores_dot.masked_fill(causal_mask == 0, float('-inf'))
        scores_euc = scores_euc.masked_fill(causal_mask == 0, float('-inf'))
        scores_liman = scores_liman.masked_fill(causal_mask == 0, float('-inf'))
        w_dot = F.softmax(scores_dot, dim=2)   # 点积权重
        w_euc = F.softmax(scores_euc, dim=2)   # 欧氏权重
        w_liman = F.softmax(scores_liman, dim=2)
        return scores_dot, scores_euc, scores_liman, w_dot, w_euc, w_liman
    def _weighted_combine(self, A_list, w, dim):

        weighted_parts = []
        for i in range(self.n_branches):

            w_i = w[..., i:i+1]

            weighted = w_i * A_list[i]   # (B, L, branch_dim[i])
            weighted_parts.append(weighted)
        return torch.cat(weighted_parts, dim=2)   # (B, L, d_state)

    def step_generate(self, u, h=None):
        """
        自回归单步生成，用于推理
        u: (B, 1, d_model)
        h: 上一时刻的隐藏状态 (B, d_state)
        """
        batch_size = u.shape[0]
        if u.dim == 2:
            u = u
        else:
            u = u.squeeze(1)
        x = self.in_proj(u.unsqueeze(1))  # (B, 1, d_state)
        u_trans = u.unsqueeze(1).transpose(1, 2)  # (B, d_model, 1)
        h1 = self.causalconv1d(u_trans)  # (B, d_state, 1)
        h1 = h1.transpose(1, 2).squeeze(1)  # (B, d_state)
        h1_expand = h1.unsqueeze(1)  # (B, 1, d_state)
        u_expand = u.unsqueeze(1)  # (B, 1, d_model)
        A_list, u_list, k_list, scale_list = self._get_branch_params(h1_expand,u_expand)
        scores_dot, scores_euc, scores_liman, w_dot, w_euc, w_liman = self._compute_scores_and_weights_generate(A_list, u_list,
                                                                                                       k_list,
                                                                                                       scale_list)

        A_dot = self._weighted_combine(A_list, w_dot, dim=-1)
        A_euc = self._weighted_combine(A_list, w_euc, dim=-1)
        A_liman = self._weighted_combine(A_list, w_liman, dim=-1)
        A = A_dot + A_euc + A_liman + A_liman * A_dot * A_euc
        B = self.B_proj(u)  # (B, L, d_state)
        C = self.C1_proj(u)  # (B, L, d_state)1
        dt = self.dt_proj(u)  # (B, L, d_state)
        K1 = self.K_proj(scores_dot)
        K2 = self.K_proj(scores_euc)
        K3 = self.K_proj(scores_liman)
        A = -torch.exp(A)
        dt = F.softplus(dt)
        I = torch.ones_like(A)
        dA = I + A * dt
        dB = B * dt
        S = I - dt
        if h is None:
            h = torch.zeros_like(A)
        h_seq1 = dA * (I - (K1 * C + K2 + K3) * S / 4) * h + (dB + K1 - K2 * K1 * K3 * C * dB) * u
        h_seq2 = dA * (I - (K2 * C + K1 + K3) * S / 4) * h + (dB + K2 - K2 * K1 * K3 * C * dB) * u
        h_seq3 = dA * (I - (K3 * C + K2 + K1) * S / 4) * h + (dB + K3 - K2 * K1 * K3 * C * dB) * u
        h_next = h_seq1 + h_seq2 + h_seq3
        h = C * h_next

        y_final = self.out_proj(h)  # (B, L, d_model)
        y_final = self.layer_norm(y_final)
        y_final = self.dropout(y_final)
        return y_final, h_next

    def combine(self, x, y):
        """
        关联扫描的组合函数
        x, y: (2, B, D_state)
        """
        m1, n1 = x[0], x[1]
        m2, n2 = y[0], y[1]
        m_comb = m2 * m1
        n_comb = m2 * n1 + n2
        return torch.stack([m_comb, n_comb], dim=0)
    def linear_scan(self, a, b):
        a = self.act(a)
        cumprod_a = torch.cumprod(a, dim=0)
        #safe_cumprod = cumprod_a + 1e-6
        s = b / (cumprod_a + 1e-12)
        prefix_sum_s = torch.cumsum(s, dim=0)
        h = cumprod_a * prefix_sum_s
        return h
    def forward(self, u):
        """
        并行前向传播，使用关联扫描
        u: (B, L, d_model)
        """
        batch_size, seq_len, _ = u.shape
        x = self.in_proj(u)                # (B, L, d_state)
        u_trans = u.transpose(1, 2)
        h = self.causalconv1d(u_trans)     # (B, d_state, L)
        h = h.transpose(1, 2)              # (B, L, d_state)
        A_list, u_list, k_list, scale_list = self._get_branch_params(h,u)
        scores_dot, scores_euc, scores_liman, w_dot, w_euc, w_liman = self._compute_scores_and_weights(A_list, u_list,
                                                                                                       k_list,
                                                                                                       scale_list)

        A_dot = self._weighted_combine(A_list, w_dot, dim=-1)
        A_euc = self._weighted_combine(A_list, w_euc, dim=-1)
        A_liman = self._weighted_combine(A_list, w_liman, dim=-1)
        A = A_dot + A_euc + A_liman + A_dot * A_liman * A_euc
        B = self.B_proj(u)  # (B, L, d_state)
        C = self.C1_proj(u)  # (B, L, d_state)
        dt = self.dt_proj(u)  # (B, L, d_state)
        K1 = self.K_proj(scores_dot)
        K2 = self.K_proj(scores_euc)
        K3 = self.K_proj(scores_liman)
        A = -torch.exp(A)
        dt = F.softplus(dt)
        I = torch.ones_like(A)
        dA = I + A * dt
        dB = B * dt
        S = I - dt
        Mt1 = dA * (I - (K1 * C + K2 + K3) * S / 4)
        Nt1 = (dB + K1 - K2 * K1 * K3 * C * dB) * u
        elems1 = torch.stack([Mt1, Nt1], dim=1).permute(2, 1, 0, 3)  # (L, 2, B, d_state)
        result1 = associative_scan(elems1, combine=self.combine)
        result1 = result1.permute(2, 1, 0, 3)  # (B, 2, L, d_state)
        h_seq1 = result1[:, 1]  # (B, L, d_state)

        Mt2 = dA * (I - (K2 * C + K1 + K3) * S / 4)
        Nt2 = (dB + K2 - K1 * K2 * K3 * C * dB) * u
        elems2 = torch.stack([Mt2, Nt2], dim=1).permute(2, 1, 0, 3)
        result2 = associative_scan(elems2, combine=self.combine)
        result2 = result2.permute(2, 1, 0, 3)
        h_seq2 = result2[:, 1]

        Mt3 = dA * (I - (K3 * C + K1 + K2) * S / 4)
        Nt3 = (dB + K3 - K1 * K2 * K3 * C * dB) * u
        elems3 = torch.stack([Mt3, Nt3], dim=1).permute(2, 1, 0, 3)
        result3 = associative_scan(elems3, combine=self.combine)
        result3 = result3.permute(2, 1, 0, 3)
        h_seq3 = result3[:, 1]
        h_next = h_seq1 + h_seq2 + h_seq3
        h = C * h_next
        y_final = self.out_proj(h)  # (B, L, d_model)
        y_final = self.layer_norm(y_final)
        y_final = self.dropout(y_final)
        return y_final