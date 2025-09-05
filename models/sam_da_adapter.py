import torch
import torch.nn as nn
import math
import os
import util.misc as utils


class SAM_DA_Adapter(nn.Module):
    def __init__(self, embed_dim, num_heads, num_adapter_tokens=64):
        super().__init__()
        
        self.num_adapter_tokens = num_adapter_tokens
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.adapter_tokens = nn.Parameter(torch.randn(1, num_adapter_tokens, embed_dim))
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.gating_param = nn.Parameter(torch.zeros(1))
        
        self.scale = self.head_dim ** -0.5

        mode = os.getenv("SAMDA_MODE", "B2").upper()
        self._apply_mode(mode)
        
    def _apply_mode(self, mode: str):
        mode = mode.upper().strip()
        self.mode = mode
        self.enabled = (mode != "B0")

        def _freeze_all_adapter_params(except_g=False):
            for n, p in self.named_parameters():
                if n.endswith("gating_param"):
                    p.requires_grad = True if except_g else False
                else:
                    p.requires_grad = False

        def _unfreeze_all():
            for _, p in self.named_parameters():
                p.requires_grad = True

        if mode == "B0":
            _freeze_all_adapter_params(except_g=False)
            if utils.is_main_process() if 'utils' in globals() else True:
                print("[SAM-DA] Mode=B0 (adapter OFF)")
        elif mode == "B1":
            _unfreeze_all()
            with torch.no_grad():
                self.gating_param.fill_(1.0)
            self.gating_param.requires_grad = False
            if utils.is_main_process() if 'utils' in globals() else True:
                print("[SAM-DA] Mode=B1 (adapter on, g=1 fixed)")
        elif mode == "B2":
            _unfreeze_all()
            if utils.is_main_process() if 'utils' in globals() else True:
                print("[SAM-DA] Mode=B2 (adapter on, g learnable)")
        elif mode == "B3":
            _freeze_all_adapter_params(except_g=True)
            if utils.is_main_process() if 'utils' in globals() else True:
                print("[SAM-DA] Mode=B3 (adapter frozen, only g learnable)")
        else:
            _unfreeze_all()
            if utils.is_main_process() if 'utils' in globals() else True:
                print(f"[SAM-DA] Unknown SAMDA_MODE={mode} → fallback to B2")
            
    def forward(self, x):
        # x는 T_l에 해당, shape: (B, M, D)
        B, M, D = x.shape
        
        # 1. 입력 피처 x를 Query로 프로젝션 (L^q)
        q = self.q_proj(x).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, n_heads, M, h_dim)
        
        # 2. 어댑터 토큰을 배치 사이즈에 맞게 확장하고, Key와 Value로 프로젝션 (L^k, L^v)
        adapter_tokens_expanded = self.adapter_tokens.expand(B, -1, -1) # (B, N, D)
        k = self.k_proj(adapter_tokens_expanded).reshape(B, self.num_adapter_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, n_heads, N, h_dim)
        v = self.v_proj(adapter_tokens_expanded).reshape(B, self.num_adapter_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, n_heads, N, h_dim)

        # 3. Scaled Dot-Product Attention 수행
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale # (B, n_heads, M, N)
        attn_probs = attn_scores.softmax(dim=-1)
        
        attn_output = (attn_probs @ v).transpose(1, 2).reshape(B, M, D) # (B, M, D)
        
        # 4. Output Projection (L^o)
        projected_output = self.out_proj(attn_output)
        
        # 5. Gating 파라미터(g_l)를 적용하고, 원래 입력 x에 더해줌 (Residual Connection)
        output = x + self.gating_param * projected_output
        
        return output
