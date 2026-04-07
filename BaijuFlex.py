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
    def __init__(self, d_model=64, d_state=1024, num_blocks=512):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.branch_dims = d_state // num_blocks
        self.n_branches = num_blocks
        self.in_proj = nn.Sequential(
            nn.Linear(d_model, d_state, bias=True),
            nn.Sigmoid(),
            nn.Linear(d_state, d_state, bias=True),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.act = nn.Sigmoid()
        self.A_projs = nn.ModuleList()
        self.u_projs = nn.ModuleList()
        self.k_projs = nn.ModuleList()
        for _ in range(self.n_branches):
            dim = self.branch_dims
            self.A_projs.append(nn.Sequential(
                nn.Linear(d_state, dim, bias=True),
                nn.Sigmoid(),
                nn.Linear(dim, dim, bias=True),
            ))
            self.u_projs.append(nn.Sequential(
                nn.Linear(d_state, dim, bias=True),
                nn.Sigmoid(),
                nn.Linear(dim, dim, bias=True),
            ))
            self.k_projs.append(nn.Sequential(
                nn.Linear(d_state, dim, bias=True),
                nn.Sigmoid(),
                nn.Linear(dim, dim, bias=True),
            ))
        self.P_chol = nn.ParameterList()
        for _ in range(self.n_branches):
            dim = self.branch_dims
            self.P_chol.append(nn.Parameter(torch.eye(dim)))
        # 因果卷积
        self.causalconv1d = CausalConv1d(
            in_channels=d_model, out_channels=d_state,
            kernel_size=2, padding=0, groups=2
        )

        # B, C, dt 投影
        self.B_proj = nn.Sequential(
            nn.Linear(d_state, d_state, bias=True),
            nn.Sigmoid(),
            nn.Linear(d_state, d_state, bias=True),
        )
        self.C1_proj = nn.Sequential(
            nn.Linear(d_state, d_state, bias=True),
            nn.Sigmoid(),
            nn.Linear(d_state, d_state, bias=True),
        )
        self.dt_proj = nn.Sequential(
            nn.Linear(d_state, d_state, bias=True),
            nn.Sigmoid(),
            nn.Linear(d_state, d_state, bias=True),
        )

        # 输出投影
        self.out_proj = nn.Linear(d_state, d_model, bias=True)

        self.dropout = nn.Dropout(p=0.2)  # 仅在输出时使用
    def _get_branch_params(self,u):

        A_list = []
        u_list = []
        k_list = []
        scale_list = []
        for i in range(self.n_branches):
            dim = self.branch_dims
            A_i = self.A_projs[i](u)      # (B, L, dim)
            u_i = self.u_projs[i](u)      # (B, L, dim)
            k_i = self.k_projs[i](u)      # (B, L, dim)
            scale_i = dim ** 0.5
            A_list.append(A_i)
            u_list.append(u_i)
            k_list.append(k_i)
            scale_list.append(scale_i)
        return A_list, u_list, k_list, scale_list

    def branch_diversity_loss(self, A_list, U_list, K_list):

        B, L, D = A_list[0].shape
        NB = self.n_branches

        stacked_A = torch.stack(A_list, dim=0)  # (NB, B, L, D)
        stacked_norm_A = F.normalize(stacked_A, p=2, dim=-1)  # (NB, B, L, D)
        gram_A = torch.einsum('n b l d, m b l d -> n m l', stacked_norm_A, stacked_norm_A)  # (NB, NB, L)
        gram_A = gram_A / (B * D)  # (NB, NB, L)
        eye_mask_A = torch.eye(NB, dtype=bool, device=gram_A.device)
        off_diag_A = gram_A[~eye_mask_A].view(NB, NB - 1, L)  # (NB, NB-1, L)
        loss_per_step_A = (off_diag_A ** 2).mean(dim=(0, 1))  # (L,)
        loss_A = loss_per_step_A.mean()  # 标量

        stacked_U = torch.stack(U_list, dim=0)  # (NB, B, L, D)
        stacked_norm_U = F.normalize(stacked_U, p=2, dim=-1)  # (NB, B, L, D)
        gram_U = torch.einsum('n b l d, m b l d -> n m l', stacked_norm_U, stacked_norm_U)  # (NB, NB, L)
        gram_U = gram_U / (B * D)  # (NB, NB, L)
        eye_mask_U = torch.eye(NB, dtype=bool, device=gram_U.device)
        off_diag_U = gram_U[~eye_mask_U].view(NB, NB - 1, L)  # (NB, NB-1, L)
        loss_per_step_U = (off_diag_U ** 2).mean(dim=(0, 1))  # (L,)
        loss_U = loss_per_step_U.mean()

        stacked_K = torch.stack(K_list, dim=0)  # (NB, B, L, D)
        stacked_norm_K = F.normalize(stacked_K, p=2, dim=-1)  # (NB, B, L, D)
        gram_K = torch.einsum('n b l d, m b l d -> n m l', stacked_norm_K, stacked_norm_K)  # (NB, NB, L)
        gram_K = gram_K / (B * D)  # (NB, NB, L)
        eye_mask_K = torch.eye(NB, dtype=bool, device=gram_K.device)
        off_diag_K = gram_K[~eye_mask_K].view(NB, NB - 1, L)  # (NB, NB-1, L)
        loss_per_step_K = (off_diag_K ** 2).mean(dim=(0, 1))  # (L,)
        loss_K = loss_per_step_K.mean()

        #gram_K_u = torch.einsum('n b l d, m b l d -> n m l', stacked_norm_K, stacked_norm_U)  # (NB, NB, L)
        #gram_K_u = gram_K_u / (B * D)  # (NB, NB, L)
        #eye_mask_K_u = torch.eye(NB, dtype=bool, device=gram_K_u.device)
        #off_diag_K_u = gram_K_u[~eye_mask_K_u].view(NB, NB - 1, L)  # (NB, NB-1, L)
        #loss_per_step_K_u = (off_diag_K_u ** 2).mean(dim=(0, 1))  # (L,)
        #loss_K_u = loss_per_step_K_u.mean()

        #gram_K_A = torch.einsum('n b l d, m b l d -> n m l', stacked_norm_K, stacked_norm_A)  # (NB, NB, L)
        #gram_K_A = gram_K_A / (B * D)  # (NB, NB, L)
        #eye_mask_K_A = torch.eye(NB, dtype=bool, device=gram_K_A.device)
        #off_diag_K_A = gram_K_A[~eye_mask_K_A].view(NB, NB - 1, L)  # (NB, NB-1, L)
        #loss_per_step_K_A = (off_diag_K_A ** 2).mean(dim=(0, 1))  # (L,)
        #loss_K_A = loss_per_step_K_A.mean()

        #gram_A_u = torch.einsum('n b l d, m b l d -> n m l', stacked_norm_A, stacked_norm_U)  # (NB, NB, L)
        #gram_A_u = gram_A_u / (B * D)  # (NB, NB, L)
        #eye_mask_A_u = torch.eye(NB, dtype=bool, device=gram_A_u.device)
        #off_diag_A_u = gram_A_u[~eye_mask_A_u].view(NB, NB - 1, L)  # (NB, NB-1, L)
        #loss_per_step_A_u = (off_diag_A_u ** 2).mean(dim=(0, 1))  # (L,)
        #loss_A_u = loss_per_step_A_u.mean()

        loss = loss_A + loss_U + loss_K
        return loss
    def _compute_scores_and_weights(self, A_list, u_list, k_list, scale_list):
        # A_list[i] shape: (B, L, dim)
        B, L, D = A_list[0].shape
        scores_dot_list = []
        scores_euc_list = []
        scores_liman_list = []
        for i in range(self.n_branches):
            dot = (A_list[i] * u_list[i] / scale_list[i] * k_list[i]).sum(dim=2, keepdim=True)  # (B, L, 1)
            scores_dot_list.append(dot)
            diff = A_list[i] - u_list[i]
            dist = torch.norm(diff, p=2, dim=2, keepdim=True)  # (B, L, 1)
            euc = -dist / (scale_list[i] + 1e-8)
            scores_euc_list.append(euc)
            L_mat = torch.tril(self.P_chol[i])  # (dim, dim)
            M_matrix = L_mat @ L_mat.T
            liman = ((diff @ M_matrix) * diff).sum(dim=2, keepdim=True)  # (B, L, 1)
            scores_liman_list.append(liman)
        scores_dot = torch.cat(scores_dot_list, dim=2)
        scores_euc = torch.cat(scores_euc_list, dim=2)
        scores_liman = torch.cat(scores_liman_list, dim=2)
        causal_mask = torch.tril(torch.ones((1, L, 1), device=scores_dot.device, dtype=scores_dot.dtype))
        scores_dot = scores_dot.masked_fill(causal_mask == 0, float('-inf'))
        scores_euc = scores_euc.masked_fill(causal_mask == 0, float('-inf'))
        scores_liman = scores_liman.masked_fill(causal_mask == 0, float('-inf'))
        loss = self.branch_diversity_loss(A_list, u_list, k_list)
        w_dot = F.softmax(scores_dot, dim=2)
        w_euc = F.softmax(scores_euc, dim=2)
        w_liman = F.softmax(scores_liman, dim=2)
        return scores_dot, scores_euc, scores_liman, w_dot, w_euc, w_liman,loss

    def _weighted_combine(self, A_list, w, dim):
        weighted_parts = []
        for i in range(self.n_branches):

            w_i = w[..., i:i+1]

            weighted = w_i * A_list[i]
            weighted_parts.append(weighted)
        return torch.cat(weighted_parts, dim=2)

    def step(self, u_init, seq_length):
        B, L, _ = u_init.shape
        N = seq_length//L + 1
        x_init = self.in_proj(u_init)  # (B, L, d_state)
        h_next = x_init
        h_total = []
        for _ in range(N):
            u = h_next
            A_list, u_list, k_list, scale_list = self._get_branch_params(u)
            scores_dot, scores_euc, scores_liman, w_dot, w_euc, w_liman, A_list_loss = self._compute_scores_and_weights(
                A_list, u_list,
                k_list,
                scale_list)
            A_dot = self._weighted_combine(A_list, w_dot, dim=-1)
            A_euc = self._weighted_combine(A_list, w_euc, dim=-1)
            A_liman = self._weighted_combine(A_list, w_liman, dim=-1)

            B = self.B_proj(u)  # (B, L, d_state)
            C = self.C1_proj(u)  # (B, L, d_state)1
            dt = self.dt_proj(u)  # (B, L, d_state)
            A_dot = -torch.exp(A_dot)
            A_euc = -torch.exp(A_euc)
            A_liman = -torch.exp(A_liman)
            dt = F.softplus(dt)
            I = torch.ones_like(A_dot)
            dA_dot = I + A_dot * dt
            dA_euc = I + A_euc * dt
            dA_liman = I + A_liman * dt
            dA = dA_dot + dA_euc + dA_liman + dA_dot * dA_euc * dA_liman
            dB = B * dt
            Mt = dA
            Nt = dB * u
            elems = torch.stack([Mt, Nt], dim=1)  # (B, 2, L, D_state)
            elems = elems.permute(2, 1, 0, 3)  # (L, 2, B, D_state)

            result = associative_scan(elems, combine=self.combine)
            result = result.permute(2, 1, 0, 3)  # (B, 2, L, D_state)
            M_prefix, N_prefix = result[:, 0], result[:, 1]  # (B, L, D_state)

            h_next = N_prefix
            h = h_next
            h = C * h
            h_total.append(h)
        y_final = torch.cat(h_total, dim=1)
        y_final = y_final[:, :seq_length, :]
        y_final = self.out_proj(y_final)  # (B, L, d_model)
        y_final = self.layer_norm(y_final)
        y_final = self.dropout(y_final)
        return y_final

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
        seq_len = a.shape[1]
        cumprod_a = torch.cumprod(a, dim=0)
        #safe_cumprod = cumprod_a + 1e-6
        s = b / (cumprod_a + 1e-12)
        prefix_sum_s = torch.cumsum(s, dim=0)
        h = cumprod_a * prefix_sum_s
        return h
    def forward(self, u_init, seq_length):
        B, L, _ = u_init.shape
        N = seq_length//L + 1
        x_init = self.in_proj(u_init)  # (B, L, d_state)
        h_next = x_init
        h_total = []
        for _ in range(N):
            u = h_next
            A_list, u_list, k_list, scale_list = self._get_branch_params(u)
            scores_dot, scores_euc, scores_liman, w_dot, w_euc, w_liman, A_list_loss = self._compute_scores_and_weights(
                A_list, u_list,
                k_list,
                scale_list)
            A_dot = self._weighted_combine(A_list, w_dot, dim=-1)
            A_euc = self._weighted_combine(A_list, w_euc, dim=-1)
            A_liman = self._weighted_combine(A_list, w_liman, dim=-1)

            B = self.B_proj(u)  # (B, L, d_state)
            C = self.C1_proj(u)  # (B, L, d_state)
            dt = self.dt_proj(u)  # (B, L, d_state)
            A_dot = -torch.exp(A_dot)
            A_euc = -torch.exp(A_euc)
            A_liman = -torch.exp(A_liman)
            dt = F.softplus(dt)
            I = torch.ones_like(A_dot)
            dA_dot = I + A_dot * dt
            dA_euc = I + A_euc * dt
            dA_liman = I + A_liman * dt
            dA = dA_dot + dA_euc + dA_liman + dA_dot * dA_euc * dA_liman
            dB = B * dt
            Mt = dA
            Nt = dB * u
            elems = torch.stack([Mt, Nt], dim=1)  # (B, 2, L, D_state)
            elems = elems.permute(2, 1, 0, 3)  # (L, 2, B, D_state)

            result = associative_scan(elems, combine=self.combine)
            result = result.permute(2, 1, 0, 3)  # (B, 2, L, D_state)
            M_prefix, N_prefix = result[:, 0], result[:, 1]  # (B, L, D_state)

            h_next = N_prefix
            h = h_next
            h = C * h
            h_total.append(h)
        y_final = torch.cat(h_total, dim=1)
        y_final = y_final[:,:seq_length,:]
        y_final = self.out_proj(y_final)  # (B, L, d_model)
        y_final = self.layer_norm(y_final)
        y_final = self.dropout(y_final)
        return y_final