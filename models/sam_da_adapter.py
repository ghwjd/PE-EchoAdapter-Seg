import torch
import torch.nn as nn
import math
import os
import util.misc as utils

# class SAM_DA_Adapter(nn.Module):
#     def __init__(self, embed_dim, num_adapter_tokens=2, attention_dim=256, num_heads=8):
#         super().__init__()
        
#         self.num_adapter_tokens = num_adapter_tokens
#         self.embed_dim = embed_dim
#         self.attention_dim = attention_dim
#         self.num_heads = num_heads
        
#         # 학습 가능한 어댑터 토큰 (A_l)
#         self.adapter_tokens = nn.Parameter(torch.randn(1, num_adapter_tokens, embed_dim))
        
#         # Single attention projections (논문의 L^q, L^k, L^v)
#         self.q_proj = nn.Linear(embed_dim, attention_dim)
#         self.k_proj = nn.Linear(embed_dim, attention_dim)
#         self.v_proj = nn.Linear(embed_dim, attention_dim)
        
#         # Output projection (L^o)
#         self.out_proj = nn.Linear(attention_dim, embed_dim)
        
#         # Final linear transformation (L^t) - 논문 수식 (5)에서 누락된 부분
#         self.final_proj = nn.Linear(embed_dim, embed_dim)
        
#         # 학습 가능한 Gating 파라미터 (g_l), 0으로 초기화
#         self.gating_param = nn.Parameter(torch.zeros(1))
        
#         self.scale = attention_dim ** -0.5

#         mode = os.getenv("SAMDA_MODE", "B2").upper()  # default B2
#         self._apply_mode(mode)


#     def _apply_mode(self, mode: str):
#         mode = mode.upper().strip()
#         self.mode = mode
#         self.enabled = (mode != "B0")

#         def _freeze_all_adapter_params(except_g=False):
#             for n, p in self.named_parameters():
#                 if n.endswith("gating_param"):
#                     p.requires_grad = True if except_g else False
#                 else:
#                     p.requires_grad = False

#         def _unfreeze_all():
#             for _, p in self.named_parameters():
#                 p.requires_grad = True

#         if mode == "B0":
#             _freeze_all_adapter_params(except_g=False)
#             if utils.is_main_process() if 'utils' in globals() else True:
#                 print("[SAM-DA] Mode=B0 (adapter OFF)")
#         elif mode == "B1":
#             _unfreeze_all()
#             with torch.no_grad():
#                 self.gating_param.fill_(1.0)
#             self.gating_param.requires_grad = False
#             if utils.is_main_process() if 'utils' in globals() else True:
#                 print("[SAM-DA] Mode=B1 (adapter on, g=1 fixed)")
#         elif mode == "B2":
#             _unfreeze_all()
#             if utils.is_main_process() if 'utils' in globals() else True:
#                 print("[SAM-DA] Mode=B2 (adapter on, g learnable)")
#         elif mode == "B3":
#             _freeze_all_adapter_params(except_g=True)
#             if utils.is_main_process() if 'utils' in globals() else True:
#                 print("[SAM-DA] Mode=B3 (adapter frozen, only g learnable)")
#         else:
#             _unfreeze_all()
#             if utils.is_main_process() if 'utils' in globals() else True:
#                 print(f"[SAM-DA] Unknown SAMDA_MODE={mode} → fallback to B2")
                

#     # def forward(self, x):
#     #     B, M, D = x.shape
        
#     #     q = self.q_proj(x).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, n_heads, M, h_dim)
        
#     #     adapter_tokens_expanded = self.adapter_tokens.expand(B, -1, -1) # (B, N, D)
#     #     k = self.k_proj(adapter_tokens_expanded).reshape(B, self.num_adapter_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, n_heads, N, h_dim)
#     #     v = self.v_proj(adapter_tokens_expanded).reshape(B, self.num_adapter_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, n_heads, N, h_dim)

#     #     attn_scores = (q @ k.transpose(-2, -1)) * self.scale # (B, n_heads, M, N)
#     #     attn_probs = attn_scores.softmax(dim=-1)
        
#     #     attn_output = (attn_probs @ v).transpose(1, 2).reshape(B, M, D) # (B, M, D)
        
#     #     projected_output = self.out_proj(attn_output)
        
#     #     output = x + self.gating_param * projected_output

#         # return output

#     def forward(self, x):
#         B, M, D = x.shape
        
#         # 1. Query projection (T_l → Q_l)
#         q = self.q_proj(x)  # (B, M, attention_dim)
        
#         # 2. Adapter tokens를 Key, Value로 projection (A_l → K_l, V_l)
#         adapter_tokens_expanded = self.adapter_tokens.expand(B, -1, -1)
#         k = self.k_proj(adapter_tokens_expanded)  # (B, N, attention_dim)
#         v = self.v_proj(adapter_tokens_expanded)  # (B, N, attention_dim)
        
#         # 3. Attention computation (수식 4)
#         attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, M, N)
#         attn_probs = attn_scores.softmax(dim=-1)
#         attn_output = attn_probs @ v  # (B, M, attention_dim)
        
#         # 4. Output projection (S_l → S'_l)
#         projected_output = self.out_proj(attn_output)  # (B, M, embed_dim)
        
#         # 5. Gating과 residual connection
#         residual_output = x + self.gating_param * projected_output
        
#         # 6. 최종 linear transformation (논문 수식 5의 Linear_t)
#         output = self.final_proj(residual_output)
        
#         return output, attn_probs

# import torch
# import torch.nn as nn
# import math
# import os
# import util.misc as utils

# class SAM_DA_Adapter(nn.Module):
#     """
#     SAM-DA Adapter (Decoder-Adapter)
#     - 기본 forward는 output만 반환
#     - return_attn=True 옵션일 때 (output, attn_probs) 반환
#     """

#     def __init__(self, embed_dim, num_heads, num_adapter_tokens=64):
#         super().__init__()
#         self.num_adapter_tokens = num_adapter_tokens
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads

#         if self.head_dim * num_heads != self.embed_dim:
#             raise ValueError("embed_dim must be divisible by num_heads")

#         # Learnable adapter tokens
#         self.adapter_tokens = nn.Parameter(torch.randn(1, num_adapter_tokens, embed_dim))

#         # Projection layers
#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.out_proj = nn.Linear(embed_dim, embed_dim)

#         # Gating parameter
#         self.gating_param = nn.Parameter(torch.zeros(1))

#         # Final projection (논문 수식 5의 Linear_t)
#         self.final_proj = nn.Linear(embed_dim, embed_dim)

#         self.scale = self.head_dim ** -0.5

#         # ---- Mode setup ----
#         mode = os.getenv("SAMDA_MODE", "B2").upper()
#         self._apply_mode(mode)

#     def _apply_mode(self, mode: str):
#         mode = mode.upper().strip()
#         self.mode = mode
#         self.enabled = (mode != "B0")

#         def _freeze_all_adapter_params(except_g=False):
#             for n, p in self.named_parameters():
#                 if n.endswith("gating_param"):
#                     p.requires_grad = True if except_g else False
#                 else:
#                     p.requires_grad = False

#         def _unfreeze_all():
#             for _, p in self.named_parameters():
#                 p.requires_grad = True

#         if mode == "B0":
#             _freeze_all_adapter_params(except_g=False)
#             if utils.is_main_process():
#                 print("[SAM-DA] Mode=B0 (adapter OFF)")
#         elif mode == "B1":
#             _unfreeze_all()
#             with torch.no_grad():
#                 self.gating_param.fill_(1.0)
#             self.gating_param.requires_grad = False
#             if utils.is_main_process():
#                 print("[SAM-DA] Mode=B1 (adapter on, g=1 fixed)")
#         elif mode == "B2":
#             _unfreeze_all()
#             if utils.is_main_process():
#                 print("[SAM-DA] Mode=B2 (adapter on, g learnable)")
#         elif mode == "B3":
#             _freeze_all_adapter_params(except_g=True)
#             if utils.is_main_process():
#                 print("[SAM-DA] Mode=B3 (adapter frozen, only g learnable)")
#         else:
#             _unfreeze_all()
#             if utils.is_main_process():
#                 print(f"[SAM-DA] Unknown SAMDA_MODE={mode} → fallback to B2")

#     def forward(self, x, return_attn=False):
#         """
#         Args:
#             x (Tensor): [B, M, D] input features
#             return_attn (bool): True일 경우 attention map도 반환
#         """
#         B, M, D = x.shape
#         # 1. Query projection
#         q = self.q_proj(x).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

#         # 2. Adapter tokens → K, V
#         adapter_tokens_expanded = self.adapter_tokens.expand(B, -1, -1)
#         k = self.k_proj(adapter_tokens_expanded).reshape(B, self.num_adapter_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = self.v_proj(adapter_tokens_expanded).reshape(B, self.num_adapter_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

#         # 3. Attention
#         attn_scores = (q @ k.transpose(-2, -1)) * self.scale
#         attn_probs = attn_scores.softmax(dim=-1)
#         attn_output = (attn_probs @ v).transpose(1, 2).reshape(B, M, D)

#         # 4. Output projection
#         projected_output = self.out_proj(attn_output)

#         # 5. Residual + Gating
#         residual_output = x + self.gating_param * projected_output

#         # 6. Final linear projection
#         output = self.final_proj(residual_output)

#         if return_attn:
#             return output, attn_probs  # (B, n_heads, M, N)
#         else:
#             return output










# sam_da_adapter.py (핵심 부분 발췌)

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