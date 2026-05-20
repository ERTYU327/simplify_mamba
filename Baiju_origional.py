import torch
import torch.nn as nn
import torch.nn.functional as F


class BaijuFlex(nn.Module):
    def __init__(self, d_model=64,d_state=16, first_block=8, other_block=16,order=1,dx=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.order = order
        self.dx = dx
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
        self.in_proj = nn.Linear(self.d_model, self.d_state, bias=True)
        self.layer_norm = nn.LayerNorm(d_model)
        self.A_projs = nn.Sequential(
            nn.Linear(d_model, d_state, bias=True),
            nn.Sigmoid(),
        )
        self.u_projs = nn.Sequential(
            nn.Linear(d_model, d_state, bias=True),
            nn.Sigmoid(),
        )
        self.k_projs = nn.Sequential(
            nn.Linear(d_model, d_state, bias=True),
            nn.Sigmoid(),
        )
        self.P_chol_raw = nn.ParameterList()
        for dim in branch_dims:
            raw = torch.zeros(dim, dim)
            self.P_chol_raw.append(nn.Parameter(raw))
        self.gamma = nn.ParameterList()
        for i in range(dx):
            self.gamma.append(nn.Parameter(torch.tensor(1.0/dx)))
        self.beta = nn.ParameterList()
        for i in range(dx):
            self.beta.append(nn.Parameter(torch.tensor(1.0/dx)))
        self.B_proj = nn.Sequential(
            nn.Linear(d_model, d_state, bias=True),
        )
        self.C1_proj = nn.Sequential(
            nn.Linear(d_model, d_state, bias=True),
            nn.Sigmoid(),
        )
        self.D_proj = nn.Sequential(
            nn.Linear(d_model, d_state, bias=True),
        )
        self.dt_proj = nn.Sequential(
            nn.Linear(d_model, d_state, bias=True),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(d_state, d_model, bias=True)
        self.K_proj = nn.ModuleList()
        for i in range(12 * order):
            self.K_proj.append(
                nn.Sequential(
                    nn.Linear(d_state, d_state, bias=True),
                    nn.Sigmoid(),
                )
            )
        self.resssm = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
        )
        self.dropout = nn.Dropout(p=0.00)
    def linear_scan(self, a, b, h0):
        eps = 1e-8
        cumprod_a = torch.cumprod(a, dim=1)
        weighted_b = b / (cumprod_a + eps)
        prefix_sum_s = torch.cumsum(weighted_b, dim=1)
        if h0 is not None:
           h = cumprod_a * (h0 + prefix_sum_s)
        else:
           h = cumprod_a * prefix_sum_s
        return h
    def _get_branch_params(self, h):
        A = self.A_projs(h)
        U = self.u_projs(h)
        k = self.k_projs(h)
        C = self.C1_proj(h)
        A_splits = torch.split(A, self.branch_dims, dim=-1)
        u_splits = torch.split(U, self.branch_dims, dim=-1)
        k_splits = torch.split(k, self.branch_dims, dim=-1)
        C_splits = torch.split(C, self.branch_dims, dim=-1)
        A_list = list(A_splits)
        u_list = list(u_splits)
        k_list = list(k_splits)
        C_list = list(C_splits)
        scale_list = [torch.sqrt(torch.tensor(dim, device=A.device, dtype=A.dtype)) for dim in self.branch_dims]
        return A_list, u_list, k_list, C_list, scale_list
    def _compute_scores(self,scores_dot,scores_euc,scores_liman):
        scores = [scores_dot,scores_euc,scores_liman,
                  scores_dot + scores_euc,scores_dot +
                  scores_liman,scores_euc + scores_liman,
                  scores_dot + scores_euc + scores_liman,
                  scores_euc * scores_dot,scores_euc *
                  scores_liman,scores_dot * scores_liman,
                  scores_dot * scores_euc * scores_liman,
                  scores_dot + scores_euc + scores_liman +
                  scores_dot * scores_euc * scores_liman]
        if self.order == 1:
            scores = scores
        else:
            for i in range(self.order - 1):
                for j in range(len(scores)):
                    scores.append(scores[j] ** (i + 1))
        return scores
    def _compute_A(self,A_dot,A_euc,A_liman):
        A_metrix = [A_dot,A_euc,A_liman,
                  A_dot + A_euc,A_dot +
                  A_liman,A_euc + A_liman,
                  A_dot + A_euc + A_liman,
                  A_euc * A_dot,A_euc *
                  A_liman,A_dot * A_liman,
                  A_dot * A_euc * A_liman,
                  A_dot + A_euc + A_liman +
                  A_dot * A_euc * A_liman]
        if self.order == 1:
            A_metrix = A_metrix
        else:
            for i in range(self.order - 1):
                for j in range(len(A_metrix)):
                    A_metrix.append(A_metrix[j] ** (i + 1))
        return A_metrix
    def _compute_K(self,scores):
        K_plus = 0
        K_cheng = 1
        K = []
        for i in range(self.order * 12):
            Ki = self.K_proj[i](scores[i])
            K_cheng = K_cheng * Ki
            K_plus = K_plus + Ki
            K.append(Ki)
        return K_cheng,K_plus,K
    def compute_A(self,A_metrix):
        A = []
        for i in range(12 * self.order):
            A_i = -torch.exp(A_metrix[i])
            A.append(A_i)
        return A
    def _compute_h_step(self,A,B,C,dt,K,K_cheng,K_plus,x,h):
        I = torch.ones_like(x)
        h_init = h
        for j in range(12 * self.order):
            h_next = 0
            h_j = h_init
            for i in range(self.dx):
                x_i = x ** (i + 1)
                dt_i = dt ** (i + 1)
                dC_i = torch.exp(((-1) ** i) * (C[j] ** (i + 1)) * dt_i)
                dA_i = torch.exp(((-1) ** i) * (A[j] ** (i + 1)) * dt_i)
                dB_i = B * dt_i
                Mt_i = dA_i * (I - (K[j] * dC_i + K_plus - K[j]) * dt_i / (12 * self.order))
                Nt_i = (dB_i + K[j] * dt_i - K_cheng * dC_i * dB_i) * x_i
                gamma_i = torch.clamp(self.gamma[i] * I, min=1e-6, max=1)
                if h_j is not None:
                   h_i = (Mt_i * h_j + Nt_i) * gamma_i
                else:
                   h_i =  Nt_i * gamma_i
                h_next = h_next + h_i
            h_init = h_next
        return h_init
    def _compute_y_step(self,C,D,dt,x,h):
        I = torch.ones_like(x)
        h_init = h
        for j in range(12 * self.order):
            h_next = 0
            h_j = h_init
            for i in range(self.dx):
                x_i = x ** (i + 1)
                dt_i = dt ** (i + 1)
                dC_i = torch.exp(((-1) ** i) * (C[j] ** (i + 1)) * dt_i)
                dD_i = D * dt_i
                Mt_i = dC_i
                Nt_i = dD_i * x_i
                beta_i = torch.clamp(self.beta[i] * I, min=1e-6, max=1)
                if h_j is not None:
                   h_i = (Mt_i * h_j + Nt_i) * beta_i
                else:
                   h_i =  Nt_i * beta_i
                h_next = h_next + h_i
            h_init = h_next
        return h_init
    def _compute_h_with_linear_scan(self,A,B,C,dt,K,K_cheng,K_plus,x,h):
        I = torch.ones_like(x)
        h_init = h
        for j in range(12 * self.order):
            h_next = 0
            h_j = h_init
            for i in range(self.dx):
                x_i = x ** (i + 1)
                dt_i = dt ** (i + 1)
                dC_i = torch.exp(((-1) ** i) * (C[j] ** (i + 1)) * dt_i)
                dA_i = torch.exp(((-1) ** i) * (A[j] ** (i + 1)) * dt_i)
                dB_i = B * dt_i
                Mt_i = dA_i * (I - (K[j] * dC_i + K_plus - K[j]) * dt_i / (12 * self.order))
                Nt_i = (dB_i + K[j] * dt_i - K_cheng * dC_i * dB_i) * x_i
                gamma_i = torch.clamp(self.gamma[i] * I, min=1e-6, max=1)
                h_i = self.linear_scan(Mt_i, Nt_i, h_j)
                h_i = gamma_i * h_i
                h_next = h_next + h_i
            h_init = h_next
        return h_init
    def _compute_y_with_linear_scan(self,C,D,dt,x,h):
        I = torch.ones_like(x)
        h_init = h
        for j in range(12 * self.order):
            h_next = 0
            h_j = h_init
            for i in range(self.dx):
                x_i = x ** (i + 1)
                dt_i = dt ** (i + 1)
                dC_i = torch.exp(((-1) ** i) * (C[j] ** (i + 1)) * dt_i)
                dD_i = D * dt_i
                Mt_i = dC_i
                Nt_i = dD_i * x_i
                beta_i = torch.clamp(self.beta[i] * I, min=1e-6, max=1)
                h_i = self.linear_scan(Mt_i, Nt_i, h_j)
                h_i = beta_i * h_i
                h_next = h_next + h_i
            h_init = h_next
        return h_init
    def _compute_sin_cos(self,dim,L,theta=10000.0):
        freqs = 1.0/(theta ** (torch.arange(0,dim,2)[:(dim//2)].float()/dim))
        t     = torch.arange(L, device=freqs.device)
        outer = torch.outer(t, freqs)
        return torch.cos(outer),torch.sin(outer)
    def rotate_half(self,x):
        x1 = x[...,:x.shape[-1]//2]
        x2 = x[...,x.shape[-1]//2:]
        return torch.cat([-x2,x1],dim=-1)
    def apply_rope_efficient(self,A,u,cos,sin):
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        cos = torch.cat((cos,cos),dim=-1)
        sin = torch.cat((sin,sin),dim=-1)
        A_rotate = self.rotate_half(A)
        u_rotate = self.rotate_half(u)
        A_embed = (A * cos) + A_rotate * sin
        u_embed = (u * cos) + u_rotate * sin
        return A_embed, u_embed
    def _compute_scores_and_weights(self, A_list, u_list, k_list, scale_list):
        B, L, D = A_list[0].shape
        scores_dot_list = []
        scores_euc_list = []
        scores_liman_list    = []
        for i, dim in enumerate(self.branch_dims):
            cos_i, sin_i = self._compute_sin_cos(dim,L)
            A_list[i], u_list[i] = self.apply_rope_efficient(A_list[i],u_list[i],cos_i,sin_i)
            dot = (A_list[i] * u_list[i] / scale_list[i] * k_list[i])
            scores_dot_list.append(dot)
            diff = A_list[i] - u_list[i]
            dist = diff**2
            euc = dist / (scale_list[i])
            scores_euc_list.append(euc)
            raw = self.P_chol_raw[i]
            H = torch.tril(raw)
            diag_indices = torch.arange(H.shape[-1], device=H.device)
            H[..., diag_indices, diag_indices] = F.softplus(H[..., diag_indices, diag_indices]) + 1e-6
            M = H @ H.T
            A_s = A_list[i][:, 0, :]
            u_s = u_list[i][:, 0, :]
            Au_s = A_s/(u_s + 1e-10)
            Au_s = torch.diag_embed(Au_s)
            M_metrix = Au_s @ M @ Au_s.transpose(1, 2)
            diff_new = diff/(A_list[i] + u_list[i] + 1e-10)
            liman = ((diff_new @ M_metrix) * diff_new)
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
        for i, dim in enumerate(self.branch_dims):
            cos_i, sin_i = self._compute_sin_cos(dim, L)
            A_list[i], u_list[i] = self.apply_rope_efficient(A_list[i], u_list[i], cos_i, sin_i)
            dot = (A_list[i] * u_list[i] / scale_list[i] * k_list[i])
            scores_dot_list.append(dot)
            diff = A_list[i] - u_list[i]
            dist = diff**2
            euc = dist / (scale_list[i])
            scores_euc_list.append(euc)
            raw = self.P_chol_raw[i]
            H = torch.tril(raw)
            diag_indices = torch.arange(H.shape[-1], device=H.device)
            H[..., diag_indices, diag_indices] = F.softplus(H[..., diag_indices, diag_indices]) + 1e-6
            M = H @ H.T
            A_s = A_list[i][:, 0, :]
            u_s = u_list[i][:, 0, :]
            Au_s = A_s / (u_s + 1e-10)
            Au_s = torch.diag_embed(Au_s)
            M_metrix = Au_s @ M @ Au_s.transpose(1, 2)
            diff_new = diff / (A_list[i] + u_list[i] + 1e-10)
            liman = ((diff_new @ M_metrix) * diff_new)
            scores_liman_list.append(liman)
        scores_dot = torch.cat(scores_dot_list, dim=2)
        scores_euc = torch.cat(scores_euc_list, dim=2)
        scores_liman = torch.cat(scores_liman_list, dim=2)
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
    def step_encoder(self, u, h):
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
        u_expand = u.unsqueeze(1)  # (B, 1, d_model)
        x = self.in_proj(u_expand)  # (B, 1, d_state)
        A_list, u_list, k_list, C_list, scale_list = self._get_branch_params(u_expand)
        scores_dot, scores_euc, scores_liman, w_dot, w_euc, w_liman = self._compute_scores_and_weights_generate(A_list, u_list,
                                                                                                       k_list,
                                                                                                       scale_list)

        A_dot = self._weighted_combine(A_list, w_dot, dim=-1)
        A_euc = self._weighted_combine(A_list, w_euc, dim=-1)
        A_liman = self._weighted_combine(A_list, w_liman, dim=-1)
        _,_,_,w_dot_c,w_euc_c,w_liman_c = self._compute_scores_and_weights_generate(C_list, u_list,k_list,scale_list)
        c_dot = self._weighted_combine(C_list, w_dot_c, dim=-1)
        c_euc = self._weighted_combine(C_list, w_euc_c, dim=-1)
        c_liman = self._weighted_combine(C_list, w_liman_c, dim=-1)
        B = self.B_proj(u)  # (B, L, d_state)
        dt = self.dt_proj(u)  # (B, L, d_state)
        scores = self._compute_scores(scores_dot,scores_euc,scores_liman)
        K_cheng,K_plus,K = self._compute_K(scores)
        A_metrix = self._compute_A(A_dot,A_euc,A_liman)
        A = self.compute_A(A_metrix)
        c_metrix = self._compute_A(c_dot,c_euc,c_liman)
        C = self.compute_A(c_metrix)
        if h is None:
            h = torch.zeros_like(A[0])
        h_next = self._compute_h_step(A, B, C, dt, K, K_cheng, K_plus, x, h)
        return h_next,C,dt,x
    def encoder(self, u, init):
        """
        并行前向传播，使用关联扫描
        u: (B, L, d_model)
        """
        if u.dim == 2:
            u = u.unsqueeze(1)
        batch_size, seq_len, _ = u.shape
        x = self.in_proj(u)                # (B, L, d_state)
        A_list, u_list, k_list, C_list, scale_list = self._get_branch_params(u)
        scores_dot, scores_euc, scores_liman, w_dot, w_euc, w_liman = self._compute_scores_and_weights(A_list, u_list,
                                                                                                       k_list,
                                                                                                       scale_list)

        A_dot = self._weighted_combine(A_list, w_dot, dim=-1)
        A_euc = self._weighted_combine(A_list, w_euc, dim=-1)
        A_liman = self._weighted_combine(A_list, w_liman, dim=-1)
        _, _, _, w_dot_c, w_euc_c, w_liman_c = self._compute_scores_and_weights_generate(C_list, u_list, k_list,scale_list)
        c_dot = self._weighted_combine(C_list, w_dot_c, dim=-1)
        c_euc = self._weighted_combine(C_list, w_euc_c, dim=-1)
        c_liman = self._weighted_combine(C_list, w_liman_c, dim=-1)
        B = self.B_proj(u)  # (B, L, d_state)
        dt = self.dt_proj(u)  # (B, L, d_state)
        scores = self._compute_scores(scores_dot, scores_euc, scores_liman)
        K_cheng, K_plus, K = self._compute_K(scores)
        A_metrix = self._compute_A(A_dot, A_euc, A_liman)
        A = self.compute_A(A_metrix)
        c_metrix = self._compute_A(c_dot, c_euc, c_liman)
        C = self.compute_A(c_metrix)
        if init is None:
            init = torch.zeros_like(A[0])
        h_next = self._compute_h_with_linear_scan(A, B, C, dt, K, K_cheng, K_plus, x, init)
        return h_next,C,dt,x
    def decoder(self,h_next,C,dt,x,u):
        D = self.D_proj(u)
        y = self._compute_y_with_linear_scan(C,D,dt,x,h_next)
        return y
    def step_decoder(self, h_next,C,dt,x,u):
        D = self.D_proj(u)
        y = self._compute_y_step(C,D,dt,x,h_next)
        return y
    def forward(self,u,init):
        u_norm = self.layer_norm(u)
        resssm = self.resssm(u)
        h_next, C, dt, x = self.encoder(u_norm,init)
        y = self.decoder(h_next,C,dt,x,u_norm)
        y = self.out_proj(y)
        y = resssm * y
        return y,h_next
    def step(self,u,h):
        u_norm = self.layer_norm(u)
        resssm = self.resssm(u)
        h_next, C, dt, x = self.step_encoder(u_norm,h)
        y = self.step_decoder(h_next,C,dt,x,u_norm)
        y = self.out_proj(y)
        y = resssm * y
        return y,h_next