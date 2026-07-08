"""Predictor modules for the Qantara world model."""

import math

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from torch.distributed.nn import all_gather as _all_gather_with_grad
from einops import rearrange
import warnings

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer."""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def _synced_randn(self, *shape, device):
        """Generate identical random tensor on all ranks (broadcast seed from rank 0)."""
        if dist.is_initialized() and dist.get_world_size() > 1:
            seed = torch.empty(1, dtype=torch.long, device=device)
            if dist.get_rank() == 0:
                seed.fill_(torch.randint(0, 2**31, (1,)).item())
            dist.broadcast(seed, src=0)
            rng = torch.Generator(device=device)
            rng.manual_seed(seed.item())
            return torch.randn(*shape, device=device, generator=rng)
        return torch.randn(*shape, device=device)

    def _gather_batch(self, proj):
        """All-gather along batch dim (1) with gradient flow. No-op on single GPU."""
        if not (dist.is_initialized() and dist.get_world_size() > 1):
            return proj
        gathered = _all_gather_with_grad(proj.contiguous())
        return torch.cat(gathered, dim=1)  # (T, B_global, D)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        proj = self._gather_batch(proj)
        A = self._synced_randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time

class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class _DINOv2Encoder(nn.Module):
    """timm DINOv2-S/14 wrapped to mimic HF ViT output (.last_hidden_state[:, 0] = CLS).
    Used by the frozen-backbone ablation; jepa.py expects HF semantics. Defined here
    (not in train.py) so torch.load(weights_only=False) can resolve the class from any
    entrypoint module — train.py's `__main__` and eval.py's `__main__` differ; pickling
    a `__main__.<class>` from train and unpickling under eval breaks otherwise."""
    def __init__(self, img_size: int):
        super().__init__()
        import timm
        from types import SimpleNamespace
        self._ns = SimpleNamespace
        self.model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True, num_classes=0, global_pool="", img_size=img_size,
        )
        self.embed_dim = self.model.embed_dim  # 384

    def forward(self, x, **_):
        return self._ns(last_hidden_state=self.model.forward_features(x))


class _ResNet18IN1kEncoder(nn.Module):
    """torchvision ResNet-18 + IN1k weights + FrozenBatchNorm2d (DETR-style stat-frozen
    BN), layer4 feature map + AdaptiveAvgPool2d → 512-d global feature. Mirrors VITA's
    image-encoder setup (resnet_observer.py:40-50) verbatim. See `_DINOv2Encoder` for
    why this lives in module.py rather than train.py."""
    def __init__(self):
        super().__init__()
        import torchvision
        from torchvision.models._utils import IntermediateLayerGetter
        from torchvision.ops.misc import FrozenBatchNorm2d
        from types import SimpleNamespace
        self._ns = SimpleNamespace
        backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
            norm_layer=FrozenBatchNorm2d,
        )
        self.features = IntermediateLayerGetter(backbone, return_layers={"layer4": "fmap"})
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embed_dim = 512

    def forward(self, x, **_):
        feat = self.pool(self.features(x)["fmap"]).flatten(1)  # (B, 512)
        return self._ns(last_hidden_state=feat.unsqueeze(1))   # (B, 1, 512)


class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        emb_dim=10,
        mlp_scale=4,
        bottleneck_dim=10,
    ):
        super().__init__()
        mlp_in_dim = input_dim if bottleneck_dim is None else bottleneck_dim
        hidden_dim = mlp_scale * emb_dim
        if hidden_dim < mlp_in_dim:
            warnings.warn("Embedder hidden_dim < input_dim: potential bottleneck")

        self.patch_embed = (
            nn.Conv1d(input_dim, bottleneck_dim, kernel_size=1, stride=1)
            if bottleneck_dim is not None
            else None
        )
        self.embed = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        if getattr(self, "patch_embed", None) is not None:
            x = self.patch_embed(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.embed(x)


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


class TimeEmbed(nn.Module):
    """DiT sinusoidal-τ + SiLU MLP for per-token flow-matching time (τ=0 noise, τ=1 clean).
    time_scale lifts τ∈[0,1] onto the DDPM range so the 10 000-period basis exercises all
    freq dims — without it, most sit near constant.
    """

    def __init__(self, dim, freq_dim=256, time_scale: float = 1000.0):
        super().__init__()
        assert freq_dim % 2 == 0, "freq_dim must be even (half cos, half sin)"
        self.time_scale = time_scale
        half = freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, tau):
        """tau: (..., ) → (..., dim). Shape-agnostic on leading dims."""
        args = (tau.float() * self.time_scale).unsqueeze(-1) * self.freqs
        return self.mlp(torch.cat([args.cos(), args.sin()], dim=-1))


class QantaraAttention(nn.Module):
    """Modality-split attention block of the Qantara predictor.

    Two SDPA passes (same Q,V; one key-mask per source
    modality), four output projections routed by (target, source) — to_out_{target}{source}.
    Only W^O_{za} (target=z ← source=a) is zero-init, so a→z coupling opens last; the
    other three start from the standard Linear init with W^O_{aa,az} scaled by 1/√2 so
    the residual variance into action targets matches the single live projection into
    state targets (asymmetric warm-start; complements the AdaLN-zero gate on QantaraBlock).

    QK-Norm (RMSNorm on per-head Q, K) bounds attention logits — DiT-stability recipe.
    RoPE (Su et al. 2021) rotates Q/K by the *block index* b(i) = i//2: both tokens in
    a block share the angle, so modality (a vs z) is decoupled from position and handled
    by the additive modality embed upstream. Half-split convention (Llama). No internal
    norm on the residual stream; QantaraBlock applies AdaLN.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, max_seq_len: int = 512,
                 rope_base: float = 10000.0):
        super().__init__()
        assert dim_head % 2 == 0, "RoPE half-split requires even dim_head"
        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.q_norm = nn.RMSNorm(dim_head)
        self.k_norm = nn.RMSNorm(dim_head)
        self.to_out_aa = nn.Linear(inner_dim, dim)
        self.to_out_az = nn.Linear(inner_dim, dim)
        self.to_out_zz = nn.Linear(inner_dim, dim)
        self.to_out_za = nn.Linear(inner_dim, dim)  # asymmetric warm-start slot (a → z)
        self.out_drop = nn.Dropout(dropout)

        # Asymmetric warm-start (DiT AdaLN-zero analog): zero-init the a→z slot, scale
        # the two live a-side projections by 1/√2 so residual-stream variance is symmetric
        # across modalities once the AdaLN gates open.
        nn.init.zeros_(self.to_out_za.weight)
        nn.init.zeros_(self.to_out_za.bias)
        with torch.no_grad():
            self.to_out_aa.weight.mul_(1.0 / math.sqrt(2.0))
            self.to_out_az.weight.mul_(1.0 / math.sqrt(2.0))

        # Precompute RoPE cos/sin keyed on block index (positions 0,0,1,1,2,2,...).
        # At block 0 the angle is 0 → RoPE is identity, which is the desired no-op at origin.
        half = dim_head // 2
        freqs = rope_base ** (-torch.arange(half, dtype=torch.float32) / half)
        block_idx = (torch.arange(max_seq_len) // 2).float()
        angles = block_idx.unsqueeze(-1) * freqs               # (max_seq_len, half)
        angles = torch.cat([angles, angles], dim=-1)           # (max_seq_len, dim_head)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, q, k):
        """Rotate (B, H, L, dim_head) Q/K by the per-position angle. (L, dim_head) broadcasts."""
        L = q.size(-2)
        cos, sin = self.rope_cos[:L], self.rope_sin[:L]
        return (q * cos + self._rotate_half(q) * sin,
                k * cos + self._rotate_half(k) * sin)

    def forward(self, x, mask_z, mask_a):
        """x: (B, L, D), pre-normalized. Even slots = action, odd = state.
        mask_{z,a}: (L, L) bool — block-causal ∧ source-modality restriction.
        Caller must include a clean prefix block (a_{-1}, z_0) so every query row
        sees ≥1 key of each modality — otherwise SDPA softmaxes over all-False and
        NaN-poisons the residual stream.
        """
        assert x.size(1) % 2 == 0, f"Qantara expects even seq len (a,z-interleaved), got L={x.size(1)}"
        drop = self.dropout if self.training else 0.0
        q, k, v = (rearrange(t, "b l (h d) -> b h l d", h=self.heads)
                   for t in self.to_qkv(x).chunk(3, dim=-1))
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = self._apply_rope(q, k)   # norm-then-rotate (Llama convention)
        # Two SDPA passes (same Q,V; key-mask filters source modality).
        out_z = rearrange(F.scaled_dot_product_attention(q, k, v, attn_mask=mask_z, dropout_p=drop),
                          "b h l d -> b l (h d)")
        out_a = rearrange(F.scaled_dot_product_attention(q, k, v, attn_mask=mask_a, dropout_p=drop),
                          "b h l d -> b l (h d)")

        y_a = self.to_out_aa(out_a[:, 0::2]) + self.to_out_az(out_z[:, 0::2])
        y_z = self.to_out_zz(out_z[:, 1::2]) + self.to_out_za(out_a[:, 1::2])
        y = torch.stack([y_a, y_z], dim=2).reshape(x.shape)
        return self.out_drop(y)


class QantaraBlock(nn.Module):
    """Transformer block of the Qantara predictor."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, max_seq_len: int = 512):
        super().__init__()
        self.attn = QantaraAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                                  max_seq_len=max_seq_len)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )
        # Pre-norms: RMSNorm without learned scale (AdaLN supplies shift+scale).
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        # Zero-init AdaLN → block is identity at step 0 (DiT recipe).
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c, mask_z, mask_a):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask_z, mask_a)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# Each mode picks (τ^a, τ^z) from four τ sources per block:
#   NOISE  = 0                  (τ=0, pure noise / bridge source)
#   CLEAN  = 1                  (τ=1, clean target)
#   VAR_A  = chain_a (∈[0,1])   (monotone cumprod chain, sampled once per row)
#   VAR_B  = chain_b (∈[0,1])   (second independent monotone chain, sampled once per row)
# Edge modes pin one modality to 0 or 1 and sweep the other via VAR_A; "joint" ties
# τ^a = τ^z (diagonal); "square" uses two independent chains (full interior).
#
# Code ↔ paper mode names (paper uses the inference-role naming):
#   cem   = paper "forward"   (latent planning at τ^a=1)
#   bc    = paper "policy"    (BC sampling at τ^z=0)
#   idm   = paper "inverse"   (inverse dynamics at τ^z=1)
#   video = paper "video"     (action-free dynamics at τ^a=0)
#   joint = paper "joint"     (diagonal co-stepping)
NOISE, CLEAN, VAR_A, VAR_B = 0, 1, 2, 3
MODE_SPECS: dict[str, tuple[int, int]] = {
    # mode:     (τ^a source, τ^z source)    [0,1]² region
    "cem":      (CLEAN,  VAR_A),           # right edge   (paper: forward)
    "bc":       (VAR_A,  NOISE),           # bottom edge  (paper: policy)
    "idm":      (VAR_A,  CLEAN),           # top edge     (paper: inverse)
    "video":    (NOISE,  VAR_A),           # left edge    (paper: video)
    "joint":    (VAR_A,  VAR_A),           # diagonal     (paper: joint)
    "square":   (VAR_A,  VAR_B),           # interior (independent chains; not in paper)
}


class Qantara(nn.Module):
    """Qantara predictor.

    Flow-matching joint predictor over next-action and next-state.

    Sequence: [a_{-1}, z_0, ã_0, z̃_1, ..., ã_{T-1}, z̃_T]; prefix block (a_{-1}, z_0)
    clean (a_{-1} is a learned start-of-sequence token). Per-example clean-prefix
    length K ∼ U{1..K_max}: target blocks t < K-1 held clean, blocks t ≥ K-1 carry a
    monotone-decreasing τ_var chain (cumprod of uniforms — PA-VDM / Rolling Forcing).
    Batch replicated M× over enabled training modes (see MODE_SPECS), each mapping
    τ_var → (τ^a, τ^z) onto a distinct edge/diagonal of [0,1]² so capacity lands on
    the inference loci rather than interior density. Loss masked to is_target.
    Attention block-causal across blocks, bidirectional within; the (target=z ←
    source=a) output projection is zero-init per layer so a→z coupling opens last.

    Three orthogonal positional signals, each handled by the mechanism best suited:
      • block index b ∈ {0..T} — RoPE on Q/K (relative, extrapolates; Llama).
      • modality ∈ {a, z}     — learned 2-vec additive embed at input.
      • flow time τ ∈ [0,1]   — AdaLN-zero per-token (DiT).

    Head parameterization (fixed):
      • z-head → x̂^z (x-prediction at τ=1) — sampled by K-step x̂-recursion (predict
        x̂, re-project onto the bridge at τ_next, repeat). K=1 is the conditional-mean
        x-prediction and is exact along the chord under γ=0. K>1 is iterative refinement
        at intermediate τ; with γ>0 + sde_rollout, intermediate steps add the
        bridge-marginal noise γ·√(τ(1−τ))·ξ so each input lies in the training marginal
        (NB: this is *not* Euler–Maruyama on the bridge SDE — the EM increment would be
        γ·√Δτ·ξ — but it keeps the model on-distribution at every τ).
      • a-head → v^a = a − ε (velocity)   — sampled by K-step Euler v-integration
        on the FM ODE.

    Three inference modes share weights:
      • rollout_z_step     — denoise z_{t+1} given clean history + candidate a_t,
                             via K state-axis x̂-recursion steps (used by CEM-style
                             planning). Supports CFG via guidance_w.
      • rollout_a_step     — denoise a_t given clean state history (τ^z pinned at 0),
                             via K action-axis Euler v-integration steps (BC sampling).
      • rollout_joint_step — denoise (a_t, z_{t+1}) jointly along the τ^a=τ^z
                             diagonal (matches "joint" training mode): action-axis
                             Euler v-integration co-stepped with state-axis x̂-recursion.
                             Used for iterative chunked BC where predicted ẑ feeds the
                             next chunk's denoising.
    """

    def __init__(
        self,
        *,
        num_frames,                       # T+1 observations; T target blocks
        depth,
        heads,
        mlp_dim,
        hidden_dim,
        embed_dim,                        # post-projector latent dim (clean z)
        action_dim,                       # raw action dim (frameskip-expanded)
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        time_freq_dim: int = 256,
        modes=("cem", "bc", "idm", "video", "joint"),  # enabled modes → batch M× replicated
        null_action_drop_p: float = 0.0,  # CFG null-action dropout probability at training (cem rows only)
        z_bridge: bool = False,           # Bridge matching on z-axis (Albergo et al. 2023). When True:
                                          #   Input:  z̃(τ_z) = (1−τ_z)·z_t + τ_z·z_{t+1} [+ Brownian noise if γ>0]
                                          #   Output: x̂^z = z_t + head(h)                  [δ-head, zero-init = identity]
                                          # CEM query at τ_z=0 sees z_t exactly — closes the noise→data
                                          # OOD gap that pure FM has at τ_z=0. At γ=0 the chord has constant
                                          # velocity and K=1 inference is exact; at γ>0 K=1 is the conditional-
                                          # mean x-prediction. a-axis remains standard FM with N(0,I) source.
        z_bridge_noise: float = 0.0,      # Brownian-bridge γ on the z-axis (only when z_bridge=True).
                                          # γ>0 adds γ·√(τ(1−τ))·ε to the linear interp (vanishes at τ∈{0,1},
                                          # peaks at 0.5) — restores K>1 expressivity for stochastic targets
                                          # without perturbing endpoint inference loci (CEM at τ_z=0, idm at 1).
        tau_chain: str = "monotone",      # Per-row τ chain shape over the T target blocks.
                                          #   monotone (default) — cumprod of per-block uniforms (PA-VDM /
                                          #                        Rolling Forcing): τ monotone along time.
                                          #   independent        — per-block U[0,1] draws (no cumprod). Ablates
                                          #                        the monotone-chain inductive bias.
        K_curriculum_max: int | None = None,  # Cap on K∼U{K_min..K_max} clean-prefix curriculum.
                                          # K=k → blocks 0..k-2 clean, blocks k-1..T-1 target (k-1 clean
                                          # transitions). None → K_max=T (full uniform). Set lower to restrict
                                          # training to the K range chunked-BC inference actually visits.
        K_curriculum_min: int = 1,        # Min for K∼U{K_min..K_max}. Default 1 = full uniform range. Set to N+1
                                          # to FORCE clean-prefix training (no K=1 wasted on no-prefix samples)
                                          # when bc_past_frames=N is used at inference. Tradeoff: model's K=1
                                          # path is never trained, so first-replan-of-episode (FIFO empty) must
                                          # use FIFO bootstrap (duplicate z_curr) at inference to stay in-dist.
        obs_flow_source: bool = False,    # VITA-style flow source. When True, eps_a is replaced by a learned
                                          # projection of z_prev (per-block obs encoding) instead of N(0,I).
                                          # Flow becomes: obs_proj → a_clean (deterministic, like VITA).
                                          # Eliminates inference-time multimodal averaging that hurts precision-
                                          # critical actions (insert/pour). At τ=0 input is obs_proj exactly;
                                          # at τ=1 input is a_clean. Costs one Linear(embed_dim, action_dim).
        z_delta: bool = True,             # Bridge-head parameterization (only when z_bridge=True). When True
                                          # (default): head outputs δ̂ and x̂ = z_t + δ̂; zero-init last layer →
                                          # identity-at-init (x̂=z_t, on-manifold prior). When False: head
                                          # outputs x̂ directly with default Linear init; tests whether the
                                          # residual skip-connection is structurally needed (vs absorbing the
                                          # z_t identity into the head's learned mapping).
    ):
        super().__init__()
        assert num_frames >= 2, "Qantara needs ≥1 target block (num_frames ≥ 2)"
        assert 0.0 <= null_action_drop_p < 1.0
        assert tau_chain in ("monotone", "independent"), f"bad tau_chain={tau_chain!r}"
        assert z_bridge_noise >= 0.0, f"z_bridge_noise must be ≥0, got {z_bridge_noise}"
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        # K-curriculum range. None K_max → uniform U{K_min..T}. Verified at forward time.
        self.K_curriculum_max = K_curriculum_max
        self.K_curriculum_min = int(K_curriculum_min)
        assert self.K_curriculum_min >= 1, f"K_curriculum_min must be ≥1, got {self.K_curriculum_min}"
        self.null_action_drop_p = null_action_drop_p
        self.z_bridge = z_bridge
        self.z_bridge_noise = z_bridge_noise
        self.z_delta = z_delta
        self.tau_chain = tau_chain

        modes = tuple(modes)
        assert len(modes) > 0, "at least one training mode must be enabled"
        unknown = [m for m in modes if m not in MODE_SPECS]
        assert not unknown, f"unknown modes: {unknown}; allowed: {sorted(MODE_SPECS)}"
        self.modes = modes
        self.register_buffer(
            "mode_spec",
            torch.tensor([MODE_SPECS[m] for m in modes], dtype=torch.long),
            persistent=False,
        )  # (M, 2): (τ^a source, τ^z source) index into [NOISE, CLEAN, VAR]

        # Modality embed [a, z]: tiled as [a, z, a, z, ...] across the interleaved sequence.
        self.modality_embed = nn.Parameter(torch.randn(1, 2, hidden_dim) * 0.02)
        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_dim))   # a_{-1} content placeholder
        self.dropout = nn.Dropout(emb_dropout)

        # Modality input projections: learnable Linear shims between encoder output and
        # the transformer trunk. a_proj_in maps raw flat action chunk (action_dim) → hidden.
        self.z_proj_in = nn.Linear(embed_dim, hidden_dim)
        self.a_proj_in = nn.Linear(action_dim, hidden_dim)
        self.time_embed = TimeEmbed(hidden_dim, freq_dim=time_freq_dim)

        # Learned null-action for CFG drop (distinct from start_token).
        self.null_action = nn.Parameter(torch.zeros(action_dim))

        # VITA-style obs-conditioned flow source. Linear(embed_dim, action_dim) projects
        # z_prev into action-latent space; output replaces eps_a in the FM interpolant.
        # zero-init bias only (preserve random Linear weight = informative obs-conditioning).
        self.obs_flow_source = obs_flow_source
        if obs_flow_source:
            self.obs_to_a_proj = nn.Linear(embed_dim, action_dim)
            nn.init.zeros_(self.obs_to_a_proj.bias)

        # RoPE max position = total tokens = 2 · num_frames (every block contributes 2 slots).
        max_seq_len = 2 * num_frames
        self.layers = nn.ModuleList([
            QantaraBlock(hidden_dim, heads, dim_head, mlp_dim, dropout, max_seq_len=max_seq_len)
            for _ in range(depth)
        ])
        # Precompute the (max_seq_len, max_seq_len) source-modality-filtered block-causal
        # masks once. _build_masks slices [:L, :L] per-call (correct because both
        # block_causal and src_is_a depend only on indices, not on L).
        idx = torch.arange(max_seq_len)
        block_causal_full = (idx // 2).unsqueeze(0) <= (idx // 2).unsqueeze(1)
        src_is_a = (idx % 2 == 0)
        self.register_buffer("mask_z_full", block_causal_full & (~src_is_a).unsqueeze(0), persistent=False)
        self.register_buffer("mask_a_full", block_causal_full & src_is_a.unsqueeze(0), persistent=False)
        self.norm = nn.RMSNorm(hidden_dim)
        # z-head: BN-MLP expander mirrors Le-WM's external pred_proj — undoes the per-token
        # RMSNorm radial constraint so the output can reach the encoder's BN-shaped target
        # distribution. Final-layer zero-init keeps identity-at-init alongside the zero
        # AdaLN gates. a-head stays Linear.
        self.z_head = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                          hidden_dim=2048, norm_fn=nn.BatchNorm1d)
        if self.z_delta:
            # δ-head + zero-init = identity-at-init via residual (x̂ = z_t + 0 = z_t).
            # With z_delta=False the residual is gone; default Linear init is the fair
            # baseline (no init-engineering confound, only structural-prior comparison).
            nn.init.zeros_(self.z_head.net[-1].weight)
            nn.init.zeros_(self.z_head.net[-1].bias)
        # a-head outputs v^a (velocity, dim=action_dim). Zero-init → no FM step at init
        # (a stays at noise; loss bootstraps from there).
        self.a_head = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.a_head.weight)
        nn.init.zeros_(self.a_head.bias)

    def __setstate__(self, state):
        """Backfill optional attributes that may be absent on older pickled ckpts.

        torch.save(model_object, ...) drops `persistent=False` buffers and any
        attributes the trained model didn't set; first access then raises
        AttributeError. Default to safe values so eval works across ckpt vintages.
        """
        super().__setstate__(state)
        defaults = {
            "obs_flow_source": False, "z_delta": True,
        }
        for k, v in defaults.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    def _z_head(self, h):
        """BN1d needs leading-dim flatten for (B*T, D)."""
        return self.z_head(h.flatten(0, -2)).unflatten(0, h.shape[:-1])

    def _sample_tau_chain(self, BM: int, T: int, is_target: torch.Tensor, device):
        """Sample one τ chain ∈ [0,1]^T per row (cumprod of per-block U[0,1] samples).
        Non-target positions (clean prefix) are pinned to 1 before the cumprod → chain
        passes through clean prefix unchanged and only varies over the target suffix.
        """
        u = torch.rand(BM, T, device=device).masked_fill(~is_target, 1.0)
        if self.tau_chain == "independent":
            # Per-block independent draws (no cumprod). Non-target blocks stay pinned to 1.
            # Ablates the monotone-chain inductive bias.
            return u
        return u.cumprod(dim=1)

    def _build_masks(self, L: int):
        """Slice the precomputed source-modality-filtered block-causal masks to L.

        Token layout: even=a slots, odd=z slots, block b(i)=i//2. Connectivity is
        block-causal across blocks ∧ bidirectional within a block. mask_z retains
        z-source keys, mask_a retains a-source keys — QantaraAttention runs two SDPA
        passes with the same Q,V and routes out_z/out_a through the four source-
        specific output projections (enables the asymmetric warm-start in
        QantaraAttention). SDPA convention: dim -1 = key axis, dim -2 = query axis.

        Lazy-rebuild guard: pickle of nn.Module drops `persistent=False` buffers, so
        ckpts saved via `torch.save(model_object, …)` lose `mask_*_full`. Re-register
        on first call after load — index-only computation, no learned state.

        Returns (mask_z, mask_a), both (L, L) bool, True = attend.
        """
        if "mask_z_full" not in self._buffers:
            device = next(self.parameters()).device
            max_seq_len = 2 * self.num_frames
            idx = torch.arange(max_seq_len, device=device)
            block_causal_full = (idx // 2).unsqueeze(0) <= (idx // 2).unsqueeze(1)
            src_is_a = (idx % 2 == 0)
            self.register_buffer("mask_z_full",
                                 block_causal_full & (~src_is_a).unsqueeze(0), persistent=False)
            self.register_buffer("mask_a_full",
                                 block_causal_full & src_is_a.unsqueeze(0), persistent=False)
        return self.mask_z_full[:L, :L], self.mask_a_full[:L, :L]

    def _transformer(self, a_tok, z_tok, tau_a, tau_z, mask_z, mask_a):
        """Interleave (a, z) tokens per block and run the backbone.

        a_tok, z_tok: (B, n, H)  per-block a- and z-tokens (block 0 = start_token + z_0).
        tau_a, tau_z: (B, n)     per-block τ for each slot.

        Block position handled by RoPE inside each layer's attention — this method only
        adds the modality embed. Sequence = 2·n tokens; block b occupies positions 2b, 2b+1.
        """
        B, n, H = a_tok.shape
        x = torch.stack([a_tok, z_tok], dim=2).reshape(B, 2 * n, H)
        x = x + self.modality_embed.repeat(1, n, 1)
        x = self.dropout(x)   # nn.Dropout is identity in eval

        tau_interleaved = torch.stack([tau_a, tau_z], dim=2).reshape(B, 2 * n)
        c = self.time_embed(tau_interleaved)

        for block in self.layers:
            x = block(x, c, mask_z, mask_a)
        return self.norm(x)

    def forward(self, z_clean, a_clean):
        """z_clean: (B, T+1, D_emb), a_clean: (B, T, D_act). Samples τ and ε internally.
        Batch is replicated M× over enabled modes (M = len(self.modes)); every row in the
        returned (B·M, ...) tensors is at position i·M + m for example i, mode m.
        Returns dict(x_z, v_a, tgt_z, tgt_a, tau_z, tau_a, is_target, mode_idx, M).
        """
        B = z_clean.size(0)
        T = z_clean.size(1) - 1        # number of target (a, z) blocks
        M = len(self.modes)
        assert z_clean.size(1) <= self.num_frames, \
            f"Qantara max num_frames={self.num_frames}, got T+1={z_clean.size(1)}"
        assert a_clean.shape == (B, T, self.action_dim), \
            f"action shape {tuple(a_clean.shape)} expected (B, {T}, {self.action_dim})"
        device = z_clean.device
        BM = B * M

        # M× replication: row i·M + m is example i rendered under mode m.
        z_clean = z_clean.repeat_interleave(M, dim=0)
        a_clean = a_clean.repeat_interleave(M, dim=0)
        mode_idx = torch.arange(M, device=device).repeat(B)   # (BM,)
        z_target = z_clean[:, 1:]

        # Clean-prefix length K ∼ U{1..K_max}; is_target flags the noisy suffix.
        # K_max defaults to T (full uniform). Cap (K_curriculum_max < T) restricts training
        # to the clean-prefix range visited at iterative chunked inference.
        K_max = T if self.K_curriculum_max is None else self.K_curriculum_max
        K_min = self.K_curriculum_min
        assert 1 <= K_min <= K_max <= T, f"K range ({K_min}..{K_max}) must be in [1, T={T}]"
        K = torch.randint(K_min, K_max + 1, (BM,), device=device)
        is_target = torch.arange(T, device=device).unsqueeze(0) >= (K - 1).unsqueeze(1)

        # Build a per-row τ-lookup table, then pick τ^a and τ^z from it by mode.
        # `sources` has shape (BM, 4, T): axis 1 indexes the 4 possible source values
        # at each (row, block): NOISE=0, CLEAN=1, VAR_A=chain_a, VAR_B=chain_b (per MODE_SPECS).
        # Fancy-indexing below: for each row i, pick the (τ^a, τ^z) source codes from
        # mode_spec[mode_idx[i]] and gather the corresponding (T,) slices into tau_{a,z}.
        # Clean-prefix blocks (not is_target) are forced back to 1 regardless of mode.
        chain_a = self._sample_tau_chain(BM, T, is_target, device)
        chain_b = self._sample_tau_chain(BM, T, is_target, device)
        sources = torch.stack([
            torch.zeros_like(chain_a),  # NOISE
            torch.ones_like(chain_a),   # CLEAN
            chain_a,                    # VAR_A
            chain_b,                    # VAR_B
        ], dim=1)
        row = torch.arange(BM, device=device)
        ones_T = torch.ones_like(chain_a)
        tau_a = torch.where(is_target, sources[row, self.mode_spec[mode_idx, 0]], ones_T)
        tau_z = torch.where(is_target, sources[row, self.mode_spec[mode_idx, 1]], ones_T)

        eps_z = torch.randn_like(z_target)
        if self.obs_flow_source:
            # VITA-style: flow source = learned projection of z_prev. Deterministic per-batch.
            # z_prev shape (B, T, embed_dim) → eps_a (B, T, action_dim).
            eps_a = self.obs_to_a_proj(z_clean[:, :-1])
        else:
            eps_a = torch.randn_like(a_clean)

        # CFG null-action substitution, gated to cem rows: CFG at inference only runs
        # through rollout_z_step (cem); null-drops in other modes would mutate v_a targets
        # toward null_action on rows that never use CFG.
        a_eff = a_clean
        if self.training and self.null_action_drop_p > 0.0 and "cem" in self.modes:
            is_cem = mode_idx == self.modes.index("cem")
            drop = (torch.rand(BM, device=device) < self.null_action_drop_p) & is_cem
            a_eff = torch.where(drop.view(BM, 1, 1), self.null_action, a_clean)

        if self.z_bridge:
            # Bridge matching: z̃(τ_z) = (1−τ_z)·z_t + τ_z·z_{t+1}. Deterministic linear
            # interpolant by default — no ε on the z-axis. At τ_z=0 input is z_t exactly
            # (CEM inference locus); at τ_z=1 input is z_{t+1} (clean target).
            # If z_bridge_noise > 0: Brownian bridge — adds γ·√(τ(1−τ))·ε term that vanishes
            # at endpoints (preserving exact CEM/idm input) and peaks at τ=0.5 (restores
            # multimodal expressivity in the middle of [0,1] for K>1 inference).
            z_prev = z_clean[:, :-1]
            z_noisy = (1.0 - tau_z.unsqueeze(-1)) * z_prev + tau_z.unsqueeze(-1) * z_target
            if self.z_bridge_noise > 0.0:
                sigma_t = self.z_bridge_noise * torch.sqrt(tau_z.unsqueeze(-1) * (1.0 - tau_z.unsqueeze(-1)))
                z_noisy = z_noisy + sigma_t * eps_z
        else:
            z_noisy = tau_z.unsqueeze(-1) * z_target + (1.0 - tau_z.unsqueeze(-1)) * eps_z

        # Action axis interpolant (FM): ã(τ) = τ·a + (1−τ)·ε, target = a − ε (velocity).
        a_noisy_token_in = tau_a.unsqueeze(-1) * a_eff + (1.0 - tau_a.unsqueeze(-1)) * eps_a
        tgt_a = a_eff - eps_a                                                  # velocity target
        tgt_z = z_target

        # Assemble T+1 blocks: block 0 = (start_token, z_0) at τ=1; blocks 1..T = (a_t, z_{t+1}) noisy.
        block0_tau = torch.ones(BM, 1, device=device)
        a_tok = torch.cat([self.start_token.expand(BM, 1, -1), self.a_proj_in(a_noisy_token_in)], dim=1)
        z_tok = self.z_proj_in(torch.cat([z_clean[:, :1], z_noisy], dim=1))
        tau_a_full = torch.cat([block0_tau, tau_a], dim=1)
        tau_z_full = torch.cat([block0_tau, tau_z], dim=1)

        mask_z, mask_a = self._build_masks(2 * (T + 1))
        h = self._transformer(a_tok, z_tok, tau_a_full, tau_z_full, mask_z, mask_a)

        # Skip block 0 (pos 0, 1); target blocks: even=ã_t, odd=z̃_{t+1}.
        h_tgt = h[:, 2:]
        # a-head outputs FM velocity v^a = head(h).
        v_a = self.a_head(h_tgt[:, 0::2])
        x_z = self._z_head(h_tgt[:, 1::2])   # clean-target prediction x̂^z
        if self.z_bridge and self.z_delta:
            # Bridge δ-parametrization: head outputs δ̂ = z_{t+1} − z_t; reconstruct x̂ = z_t + δ̂.
            # With zero-init head, x̂(init) = z_t — identity-at-init prior. Loss form
            # ‖x̂ − z_{t+1}‖² = ‖head(h) − δ‖² so the head supervision is on the residual.
            # When z_delta=False, head outputs x̂ directly (loss = ‖head − z_{t+1}‖²).
            x_z = z_clean[:, :-1] + x_z
        out = {"x_z": x_z, "v_a": v_a, "tgt_z": tgt_z, "tgt_a": tgt_a,
               "tau_z": tau_z, "tau_a": tau_a, "is_target": is_target,
               "mode_idx": mode_idx, "M": M,
               "eps_a": eps_a}
        return out

    def rollout_z_step(self, z_hist, a_hist, a_current, K: int = 1, eps_z=None,
                       guidance_w: float = 1.0):
        """K-step state-axis x̂-recursion for z_{t+1} given clean history + candidate a_t.
        Used by CEM-style planning: the planner queries this with each candidate a_t.

        τ^a=1 (action pinned), τ^z swept 0→1 in K steps. The z-head is x-pred, so each
        step predicts x̂ and re-projects onto the bridge (or FM chord) at τ_next — DDIM-
        style x-prediction recursion, not Euler v-integration. K=1 is the single
        x-prediction call and is exact along the chord under γ=0.

        With γ=`z_bridge_noise`>0 and K>1 we additionally inject bridge-marginal noise
        γ·√(τ_next·(1−τ_next))·ξ at each intermediate τ_next ∈ (0,1) — matches the
        training-time bridge marginal so the iterative input is in-distribution. γ=0
        and the τ_next=1 endpoint are no-ops (σ=0). NB: marginal resampling, not true
        Euler–Maruyama (EM increment would be γ·√Δτ·ξ); we sample from the marginal
        directly to keep every iterate on the training manifold.

        z_hist     : (B, t+1, D_emb)  clean z_0..z_t
        a_hist     : (B, t,   D_act)  clean a_0..a_{t-1}
        a_current  : (B,      D_act)  candidate a_t
        eps_z      : optional (B, D_emb); default fresh N(0, I).
        guidance_w : CFG scale. w=1 bypasses; w>1 adds a null-action forward per step,
                     x_null + w·(x_cond − x_null). Requires null_action_drop_p>0 at train.
        Returns    : ẑ_{t+1}, shape (B, D_emb).
        """
        B, num_hist, D_emb = z_hist.shape   # num_hist = t+1 clean states z_0..z_t
        n_blocks = num_hist + 1             # + 1 new (a_t, z_{t+1}) block to denoise
        use_cfg = guidance_w != 1.0

        assert a_hist.shape == (B, num_hist - 1, self.action_dim)
        assert a_current.shape == (B, self.action_dim)
        assert 2 * n_blocks <= 2 * self.num_frames, \
            f"rollout L={2 * n_blocks} > max {2 * self.num_frames} (num_frames={self.num_frames})"
        assert not use_cfg or self.null_action_drop_p > 0.0, \
            "CFG (guidance_w != 1) requires training with null_action_drop_p > 0"

        device = z_hist.device
        # BN in z_head would drift on non-i.i.d. rollout batch; restore mode on exit.
        was_training = self.training
        self.eval()

        a_all = torch.cat([a_hist, a_current.unsqueeze(1)], dim=1)   # a_0..a_t  (raw)
        a_tok_cond = torch.cat(
            [self.start_token.expand(B, 1, -1), self.a_proj_in(a_all)], dim=1,
        )
        z_clean_tok = self.z_proj_in(z_hist)
        mask_z, mask_a = self._build_masks(2 * n_blocks)
        tau_a_seq = torch.ones(B, n_blocks, device=device)   # τ^a=1 throughout (Regime B)
        tau_z_seq = torch.ones(B, n_blocks, device=device)

        if use_cfg:
            null_proj = self.a_proj_in(self.null_action).view(1, 1, -1).expand(B, num_hist, -1)
            a_tok_null = torch.cat([self.start_token.expand(B, 1, -1), null_proj], dim=1)

        eps = eps_z if eps_z is not None else torch.randn(B, D_emb, device=device)
        taus = torch.linspace(0.0, 1.0, K + 1, device=device)
        z_last = z_hist[:, -1]
        # Bridge: input at τ=0 is z_last exactly (no target estimate yet to interpolate toward).
        # FM: input at τ=0 is pure ε.
        z_noisy = z_last if self.z_bridge else eps

        for tau, tau_next in zip(taus[:-1], taus[1:]):
            z_tok = torch.cat([z_clean_tok, self.z_proj_in(z_noisy.unsqueeze(1))], dim=1)
            tau_z_seq[:, -1] = tau

            h = self._transformer(a_tok_cond, z_tok, tau_a_seq, tau_z_seq, mask_z, mask_a)
            x_hat = self._z_head(h[:, -1])
            if use_cfg:
                h_null = self._transformer(a_tok_null, z_tok, tau_a_seq, tau_z_seq, mask_z, mask_a)
                x_null = self._z_head(h_null[:, -1])
                x_hat = x_null + guidance_w * (x_hat - x_null)
            if self.z_bridge:
                # Head outputs δ̂ (z_delta=True) or x̂ directly (z_delta=False). Absolute
                # prediction is x̂ = z_last + δ̂ (with residual) or x̂ = head (without).
                # Re-project onto the bridge at τ_next: z̃ = (1−τ_next)·z_last + τ_next·x̂.
                # Under γ=0 the chord has constant velocity → K>1 is degenerate (re-projection
                # rolls back to the same x̂); K=1 is the exact one-call solution.
                if self.z_delta:
                    x_hat = z_last + x_hat
                z_noisy = (1.0 - tau_next) * z_last + tau_next * x_hat
                # Bridge-marginal noise injection at intermediate steps: σ(τ_next) =
                # γ·√(τ_next·(1−τ_next)) matches the *training-time* marginal at τ_next
                # (with x̂ standing in for z_{t+1}). γ=0 → σ=0 (no-op for FM/linear-chord
                # bridges); τ_next=1 endpoint → σ=0 (final estimate unperturbed); τ_next=0
                # not entered. NB: marginal resampling, not Euler–Maruyama on the bridge SDE.
                if self.z_bridge_noise > 0.0 and 0.0 < float(tau_next) < 1.0:
                    sigma = self.z_bridge_noise * torch.sqrt(tau_next * (1.0 - tau_next))
                    z_noisy = z_noisy + sigma * torch.randn_like(z_noisy)
            else:
                # Re-project onto FM trajectory at τ_next.
                z_noisy = (1.0 - tau_next) * eps + tau_next * x_hat

        if was_training:
            self.train()
        return z_noisy

    def rollout_a_step(self, z_hist, a_hist, K: int = 1, eps_a=None):
        """Pure-BC action-denoise rollout. K-step FM v-integration of the a-axis, τ^a: 0→1.

        a_noisy starts at ε ~ N(0,I); each step updates a += dτ·v̂.

        τ^z pinned at 0 across all k (matches the "bc" training locus, (VAR_A, NOISE)) —
        z-side is *not* denoised in this loop. Bridge convention: input at τ_z=0 collapses
        to z_prev; FM convention: input is ε.

        z_hist : (B, t+1, D_emb) clean z_0..z_t
        a_hist : (B, t, D_act)   clean a_0..a_{t-1}  (t may be 0)
        Returns: â_t, shape (B, D_act).
        """
        B, num_hist, D_emb = z_hist.shape   # num_hist = t+1 clean states z_0..z_t
        n_blocks = num_hist + 1             # + 1 new (a_t, z_{t+1}) block to denoise

        assert a_hist.shape == (B, num_hist - 1, self.action_dim)
        assert 2 * n_blocks <= 2 * self.num_frames, \
            f"rollout L={2 * n_blocks} > max {2 * self.num_frames} (num_frames={self.num_frames})"

        device = z_hist.device

        # BN in z_head unused here (a-head only); eval-guard for parity + future-proofing.
        was_training = self.training
        self.eval()

        # z-side fixed across k: blocks 0..num_hist-1 clean, block num_hist at τ_z=0.
        # Bridge: input collapses to z_prev. FM: input is ε.
        if self.z_bridge:
            z_curr = z_hist[:, -1]
        else:
            z_curr = torch.randn(B, D_emb, device=device)
        z_tok = self.z_proj_in(torch.cat([z_hist, z_curr.unsqueeze(1)], dim=1))
        a_hist_tok = self.a_proj_in(a_hist)   # (B, num_hist-1, H); Linear handles length 0

        mask_z, mask_a = self._build_masks(2 * n_blocks)
        tau_a_seq = torch.ones(B, n_blocks, device=device)
        tau_z_seq = torch.ones(B, n_blocks, device=device)
        tau_z_seq[:, -1] = 0.0

        # FM v-integration on the a-axis.
        if eps_a is not None:
            eps = eps_a
        elif self.obs_flow_source:
            # VITA-style: deterministic obs-derived flow source. z_hist[:, -1] is z_t.
            eps = self.obs_to_a_proj(z_hist[:, -1])
        else:
            eps = torch.randn(B, self.action_dim, device=device)
        a_noisy = eps
        dtau = 1.0 / K
        for k in range(K):
            a_tok = torch.cat([
                self.start_token.expand(B, 1, -1),
                a_hist_tok,
                self.a_proj_in(a_noisy.unsqueeze(1)),
            ], dim=1)
            tau_a_seq[:, -1] = k * dtau
            h = self._transformer(a_tok, z_tok, tau_a_seq, tau_z_seq, mask_z, mask_a)
            a_noisy = a_noisy + dtau * self.a_head(h[:, -2])   # current-block a-slot
        a_out = a_noisy

        if was_training:
            self.train()
        return a_out

    def rollout_joint_step(self, z_hist, a_hist, K: int = 1, eps_a=None,
                           return_z: bool = False):
        """Joint BC inference: simultaneously denoise (a_t, z_{t+1}) along the diagonal
        τ^a=τ^z (matches the "joint" training mode, (VAR_A, VAR_A)). Each step does one
        action-axis Euler v-integration update co-stepped with one state-axis x̂-recursion
        update at the same τ.

        Returns â_t shape (B, D_act) by default. If `return_z=True`, returns
        (â_t, ẑ_{t+1}) — needed for iterative chunked BC inference where the predicted
        z is fed forward as clean context for the next chunk's denoising. Default False
        keeps single-chunk callers (jepa.JEPA.get_action) compatible — they replace ẑ
        with the true env z after env.step().

        CFG is intentionally not exposed here — joint BC has no use case for action
        guidance (CFG only makes sense in CEM where we score actions). Use
        rollout_z_step if guidance_w is needed.

        Contrast with rollout_a_step (τ^z=NOISE pinned): there z is held fixed at z_prev
        (bridge) or ε (FM) across all K steps and only τ^a moves. Joint sweeps both,
        letting the action prediction co-condition on the model's refining estimate of
        the next state.
        """
        B, num_hist, D_emb = z_hist.shape
        n_blocks = num_hist + 1
        assert a_hist.shape == (B, num_hist - 1, self.action_dim)
        assert 2 * n_blocks <= 2 * self.num_frames, \
            f"rollout L={2 * n_blocks} > max {2 * self.num_frames} (num_frames={self.num_frames})"

        device = z_hist.device
        was_training = self.training
        self.eval()

        a_hist_tok = self.a_proj_in(a_hist)
        z_clean_tok = self.z_proj_in(z_hist)
        mask_z, mask_a = self._build_masks(2 * n_blocks)
        tau_a_seq = torch.ones(B, n_blocks, device=device)
        tau_z_seq = torch.ones(B, n_blocks, device=device)

        # τ=0 init: z-side bridge → z_prev / FM → ε. a-side: ε.
        z_last = z_hist[:, -1]
        if self.z_bridge:
            z_noisy = z_last
        else:
            z_eps = torch.randn(B, D_emb, device=device)
            z_noisy = z_eps
        if eps_a is not None:
            eps = eps_a
        elif self.obs_flow_source:
            eps = self.obs_to_a_proj(z_hist[:, -1])
        else:
            eps = torch.randn(B, self.action_dim, device=device)
        a_noisy = eps

        taus = torch.linspace(0.0, 1.0, K + 1, device=device)
        for tau, tau_next in zip(taus[:-1], taus[1:]):
            a_tok = torch.cat([
                self.start_token.expand(B, 1, -1),
                a_hist_tok,
                self.a_proj_in(a_noisy.unsqueeze(1)),
            ], dim=1)
            z_tok = torch.cat([z_clean_tok, self.z_proj_in(z_noisy.unsqueeze(1))], dim=1)
            tau_a_seq[:, -1] = tau
            tau_z_seq[:, -1] = tau

            h = self._transformer(a_tok, z_tok, tau_a_seq, tau_z_seq, mask_z, mask_a)

            # a-update: Euler v-integration (FM).
            dtau = float(tau_next - tau)
            a_noisy = a_noisy + dtau * self.a_head(h[:, -2])

            # z-update: one x̂-recursion step. Bridge → x̂ = z_last + δ̂ (z_delta=True) or
            # x̂ = head (z_delta=False), re-project onto the bridge at τ_next; FM → x̂ direct,
            # re-project onto the FM chord. Bridge-marginal noise injection mirrors
            # rollout_z_step: γ>0 + intermediate τ_next in (0,1) → add γ·√(τ(1−τ))·ξ to keep
            # iterative inputs on the training-time marginal. γ=0 / endpoints → no-op.
            x_z = self._z_head(h[:, -1])
            if self.z_bridge:
                x_hat_z = z_last + x_z if self.z_delta else x_z
                z_noisy = (1.0 - tau_next) * z_last + tau_next * x_hat_z
                if self.z_bridge_noise > 0.0 and 0.0 < float(tau_next) < 1.0:
                    sigma = self.z_bridge_noise * torch.sqrt(tau_next * (1.0 - tau_next))
                    z_noisy = z_noisy + sigma * torch.randn_like(z_noisy)
            else:
                z_noisy = (1.0 - tau_next) * z_eps + tau_next * x_z

        if was_training:
            self.train()
        a_out = a_noisy
        if return_z:
            return a_out, z_noisy
        return a_out

    def rollout_video_step(self, z_hist, a_hist):
        """Single forward pass at the video locus (τ^a=0, τ^z=0): predict ẑ_{t+1} with
        the action input set to pure ε. The action token carries no information about
        a_t, so the network's output approximates E[z_{t+1} | z_hist] marginalised over
        the action prior. Mirrors rollout_z_step at K=1 with action ε instead of clean.

        z_hist : (B, t+1, D_emb) clean z_0..z_t
        a_hist : (B, t,   D_act) clean a_0..a_{t-1}
        Returns: ẑ_{t+1}, shape (B, D_emb).
        """
        B, num_hist, D_emb = z_hist.shape
        n_blocks = num_hist + 1
        assert a_hist.shape == (B, num_hist - 1, self.action_dim)
        assert 2 * n_blocks <= 2 * self.num_frames, \
            f"rollout L={2 * n_blocks} > max {2 * self.num_frames} (num_frames={self.num_frames})"

        device = z_hist.device
        was_training = self.training
        self.eval()

        # τ^z[-1]=0 source: bridge → z_prev exactly; FM → ε.
        z_last = z_hist[:, -1]
        z_noisy = z_last if self.z_bridge else torch.randn(B, D_emb, device=device)
        z_tok = self.z_proj_in(torch.cat([z_hist, z_noisy.unsqueeze(1)], dim=1))

        # τ^a[-1]=0 source: ε.
        a_hist_tok = self.a_proj_in(a_hist)
        a_in_tok = torch.randn(B, self.action_dim, device=device)
        a_tok = torch.cat([
            self.start_token.expand(B, 1, -1),
            a_hist_tok,
            self.a_proj_in(a_in_tok.unsqueeze(1)),
        ], dim=1)

        mask_z, mask_a = self._build_masks(2 * n_blocks)
        tau_a_seq = torch.ones(B, n_blocks, device=device)
        tau_z_seq = torch.ones(B, n_blocks, device=device)
        tau_a_seq[:, -1] = 0.0
        tau_z_seq[:, -1] = 0.0

        h = self._transformer(a_tok, z_tok, tau_a_seq, tau_z_seq, mask_z, mask_a)
        x_hat = self._z_head(h[:, -1])
        if self.z_bridge and self.z_delta:
            x_hat = z_last + x_hat

        if was_training:
            self.train()
        return x_hat

    def rollout_idm_step(self, z_hist, a_hist, z_target, K: int = 1, eps_a=None):
        """K-step Euler on the action axis at the idm locus (τ^a swept, τ^z=1):
        denoise â_t given clean (z_hist, z_target). Mirrors rollout_a_step but with
        z_target as the last-block z input and τ^z[-1]=1 (= idm training locus).

        z_hist   : (B, t+1, D_emb) clean z_0..z_t
        a_hist   : (B, t,   D_act) clean a_0..a_{t-1}
        z_target : (B,      D_emb) clean ẑ_{t+1} (e.g. from rollout_video_step)
        Returns  : â_t, shape (B, D_act).
        """
        B, num_hist, D_emb = z_hist.shape
        n_blocks = num_hist + 1
        assert a_hist.shape == (B, num_hist - 1, self.action_dim)
        assert z_target.shape == (B, D_emb)
        assert 2 * n_blocks <= 2 * self.num_frames, \
            f"rollout L={2 * n_blocks} > max {2 * self.num_frames} (num_frames={self.num_frames})"

        device = z_hist.device
        was_training = self.training
        self.eval()

        # Last z slot = clean target (τ^z[-1]=1, the idm locus).
        z_tok = self.z_proj_in(torch.cat([z_hist, z_target.unsqueeze(1)], dim=1))
        a_hist_tok = self.a_proj_in(a_hist)

        mask_z, mask_a = self._build_masks(2 * n_blocks)
        tau_a_seq = torch.ones(B, n_blocks, device=device)
        tau_z_seq = torch.ones(B, n_blocks, device=device)
        # tau_z_seq[:, -1] is already 1.0 — idm locus.

        if eps_a is not None:
            eps = eps_a
        elif self.obs_flow_source:
            eps = self.obs_to_a_proj(z_hist[:, -1])
        else:
            eps = torch.randn(B, self.action_dim, device=device)
        a_noisy = eps
        dtau = 1.0 / K
        for k in range(K):
            a_tok = torch.cat([
                self.start_token.expand(B, 1, -1),
                a_hist_tok,
                self.a_proj_in(a_noisy.unsqueeze(1)),
            ], dim=1)
            tau_a_seq[:, -1] = k * dtau
            h = self._transformer(a_tok, z_tok, tau_a_seq, tau_z_seq, mask_z, mask_a)
            a_noisy = a_noisy + dtau * self.a_head(h[:, -2])
        a_out = a_noisy

        if was_training:
            self.train()
        return a_out

    def rollout_video_idm_step(self, z_hist, a_hist, K: int = 1, eps_a=None):
        """Composed video → idm inference: a third inference path on a single trained
        checkpoint, alongside CEM forward (rollout_z_step) and BC policy (rollout_a_step).

        1. video predicts ẑ_{t+1} ≈ E[z_{t+1} | z_hist] action-blind (τ^a=0, τ^z=0).
        2. idm denoises â_t given clean (z_hist, ẑ_{t+1}) (τ^a swept, τ^z=1).

        K controls the idm action-axis Euler step count; video is a single x-prediction
        call. Action-marginal model-based imitation: imitate the data dynamics first,
        then extract the action consistent with the predicted transition.
        """
        z_pred = self.rollout_video_step(z_hist, a_hist)
        return self.rollout_idm_step(z_hist, a_hist, z_pred, K=K, eps_a=eps_a)


class ImageDecoder(nn.Module):
    """FC → (spatial, spatial, base_channels) → Upsample+Conv stack → RGB pixels.
    Upsample+Conv (not ConvTranspose2d) avoids checkerboard artifacts. n_up targets a
    ~8×8 spatial start; channels halve each stage, floored at 16.
    """

    def __init__(self, embed_dim: int, img_size: int = 64, base_channels: int = 128):
        super().__init__()
        n_up = max(2, round(math.log2(img_size / 8)))
        spatial = img_size // (2 ** n_up)
        assert spatial * (2 ** n_up) == img_size, \
            f"img_size={img_size} not exactly divisible by 2^{n_up}"
        self.spatial = spatial
        self.ch = base_channels
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, base_channels * spatial * spatial),
            nn.GELU(),
        )
        layers, ch = [], base_channels
        for _ in range(n_up):
            ch_next = max(16, ch // 2)
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(ch, ch_next, 3, padding=1),
                nn.GroupNorm(math.gcd(ch_next, 8), ch_next),
                nn.GELU(),
            ]
            ch = ch_next
        layers.append(nn.Conv2d(ch, 3, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z).view(z.size(0), self.ch, self.spatial, self.spatial)
        return self.net(x)


class WarmupStableDecay(torch.optim.lr_scheduler._LRScheduler):
    """WSD: linear warmup → constant peak → linear decay to eta_min.

    Refs: Hu et al., MiniCPM (arXiv:2404.06395, 2024); river-valley landscape view
    (Wen et al., ICLR 2025). Decay-to-zero (D2Z) is the default (eta_min=0.0).
    Stepped per-batch under spt's manual optimization, like LinearWarmupCosineAnnealingLR.
    """

    def __init__(self, optimizer, warmup_steps, stable_steps, decay_steps,
                 warmup_start_lr=0.0, eta_min=0.0, last_epoch=-1):
        self.warmup_steps = int(warmup_steps)
        self.stable_steps = int(stable_steps)
        self.decay_steps = int(decay_steps)
        self.warmup_start_lr = float(warmup_start_lr)
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        s = self.last_epoch
        decay_start = self.warmup_steps + self.stable_steps
        end = decay_start + self.decay_steps
        out = []
        for base_lr in self.base_lrs:
            if s < self.warmup_steps:
                lr = self.warmup_start_lr + (base_lr - self.warmup_start_lr) * s / max(1, self.warmup_steps)
            elif s < decay_start:
                lr = base_lr
            elif s < end:
                t = (s - decay_start) / max(1, self.decay_steps)
                lr = self.eta_min + (base_lr - self.eta_min) * (1 - t)
            else:
                lr = self.eta_min
            out.append(lr)
        return out
