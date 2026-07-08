"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from module import Qantara

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        decoder=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.decoder = decoder

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info and self.action_encoder is not None:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """
        if isinstance(self.predictor, Qantara):
            return self._rollout_qantara(info, action_sequence, history_size)
        return self._rollout_lewm(info, action_sequence, history_size)

    def _rollout_lewm(self, info, action_sequence, history_size: int = 3):
        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def _rollout_qantara(self, info, action_sequence, history_size=None):
        """Qantara rollout for planning eval: sliding-window z-only autoregression with
        K state-axis x̂-recursion steps per env step. All history z's treated as clean;
        candidate action pinned at τ^a=1 (the planner-side conditioning locus).
        Output shape matches Le-WM: (B, S, T+1, D_emb).

        history_size is ignored — Qantara uses the predictor's full num_frames window
        (swm.policy doesn't plumb cfg.wm.history_size through get_cost → rollout).
        K / guidance_w read from self.rollout_{k,guidance_w}, set in train.py.
        """
        del history_size
        assert "pixels" in info, "pixels not in info_dict"
        predictor = self.predictor
        K = getattr(self, "rollout_k", 1)
        guidance_w = getattr(self, "rollout_guidance_w", 1.0)
        B, S, T = action_sequence.shape[:3]
        HS = predictor.num_frames - 1  # clean-history cap; 1 slot reserved for target block

        # copy and encode initial info dict (stub action — never reaches the predictor)
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init["action"] = action_sequence[:, 0, :1]
        _init = self.encode(_init)
        info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(info["emb"], "b s t d -> (b s) t d").clone()
        act = rearrange(action_sequence, "b s t a -> (b s) t a")

        D = emb.size(-1)
        n_steps = T - emb.size(1) + 1
        for _ in range(n_steps):
            cur = emb.size(1)
            hs = min(cur, HS)
            start = cur - hs
            z_win = emb[:, start:cur]        # (BS, hs, D)
            a_win = act[:, start:cur]        # a_{start..cur-1}; last = candidate "current" action
            # CRN variance reduction: one ε per (start,goal) shared across the S CEM
            # candidates. Per-candidate noise lets CEM ranking be dominated by sampling
            # noise in small-action-signal regimes (pusht).
            eps_z = torch.randn(B, 1, D, device=emb.device).expand(B, S, D).reshape(B * S, D)
            z_next = predictor.rollout_z_step(
                z_win, a_win[:, :-1], a_win[:, -1],
                K=K, guidance_w=guidance_w, eps_z=eps_z,
            )
            emb = torch.cat([emb, z_next.unsqueeze(1)], dim=1)

        # unflatten batch and sample dimensions
        info["predicted_emb"] = rearrange(emb, "(b s) t d -> b s t d", b=B, s=S)
        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_action(self, info_dict: dict, **kwargs):
        """BC inference path. Qantara-only. swm.policy.FeedForwardPolicy calls this once per
        env step; we predict the full frameskip-block once and buffer the per-frame slice
        across env steps so each call returns one (E, action_dim_raw).

        info_dict (post-`_prepare_info`):
            pixels: (E, T, C, H, W) — E envs × T history frames; C=3, H=W=224.
            (goal/proprio/state present but unused — BC mode is not goal-conditioned.)

        The predictor outputs (E, frameskip × action_dim_raw); env.step expects
        (E, action_dim_raw). action_dim_raw is read from `self.action_dim_raw` (set in
        train.py at training time). If absent on the pickled object, we assume no
        frameskip-stacking (action_dim_raw == predictor.action_dim).

        Open-loop, no goal conditioning — by construction, this is the BC counterpart
        to the goal-aware CEM path through the same trained checkpoint.
        """
        assert isinstance(self.predictor, Qantara), \
            "JEPA.get_action is Qantara-only (BC dispatch); LeWM has no rollout_a_step path."
        assert "pixels" in info_dict, "BC dispatch requires 'pixels' in info_dict"
        # Pop pre-computed frame from buffer if available — keeps each env step cheap.
        if getattr(self, "_bc_buffer", None):
            return self._bc_buffer.pop(0)

        device = next(self.parameters()).device
        info = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in info_dict.items()}

        if "action" not in info or info.get("action") is None:
            E_, T_ = info["pixels"].shape[:2]
            info["action"] = torch.zeros(E_, T_, self.predictor.action_dim, device=device)
        info = self.encode(info)
        z_curr = info["emb"][:, -1:]                            # (E, 1, D) — current frame only
        E = z_curr.size(0)

        # Frame-spaced past-context. bc_past_frames=0 (default) → single-frame BC:
        # z_hist=(E,1,D), a_hist empty. bc_past_frames=K>0 → maintain a frame-level FIFO
        # of past (z, action_chunk) pairs, updated once per model invocation (= once per
        # frameskip env steps via _bc_buffer below). The FIFO is FRAME-spaced by
        # construction, matching the training data convention. Capped at num_frames-1.
        n_past_frames = int(getattr(self, "bc_past_frames", 0))
        # n_blocks = (n_past_frames + 1 current) + 1 query = n_past_frames + 2 ≤ num_frames.
        max_past = self.predictor.num_frames - 2
        if n_past_frames > max_past:
            n_past_frames = max_past
        if n_past_frames > 0:
            if not hasattr(self, "_bc_z_history"):
                self._bc_z_history = []
            if not hasattr(self, "_bc_a_history"):
                self._bc_a_history = []
            z_past = self._bc_z_history[-n_past_frames:]
            a_past = self._bc_a_history[-n_past_frames:]
            z_hist = torch.cat(z_past + [z_curr], dim=1) if z_past else z_curr
            if a_past:
                a_hist = torch.stack(a_past, dim=1)
            else:
                a_hist = torch.zeros(E, 0, self.predictor.action_dim, device=device)
            # Invariant: a_hist length = z_hist length - 1 (k past actions sit between k+1 z's).
            assert a_hist.size(1) == z_hist.size(1) - 1
        else:
            z_hist = z_curr
            a_hist = torch.zeros(E, 0, self.predictor.action_dim, device=device)

        K = getattr(self, "rollout_a_k", 1)
        # bc_kind ∈ {"bc", "joint"}. "bc" = pure single-modality denoise (τ^z=NOISE pinned;
        # for z_bridge models that's z_prev). "joint" = diagonal co-stepping (τ^a=τ^z swept
        # together) matching the "joint" training mode.
        bc_kind = getattr(self, "bc_inference_kind", "bc")
        if bc_kind == "joint":
            action_concat = self.predictor.rollout_joint_step(z_hist, a_hist, K=K)
        elif bc_kind == "video_idm":
            # Third inference path: video predicts ẑ_{t+1} action-blind, then idm denoises
            # â_t given the predicted target. K controls idm's action-axis Euler step count;
            # video is always a single x-prediction call.
            action_concat = self.predictor.rollout_video_idm_step(z_hist, a_hist, K=K)
        else:
            action_concat = self.predictor.rollout_a_step(z_hist, a_hist, K=K)

        # Update frame-level deques after the prediction (so the next invocation's a_hist
        # contains exactly the chunk that's about to be executed).
        if n_past_frames > 0:
            self._bc_z_history.append(z_curr.detach())
            self._bc_a_history.append(action_concat.detach())

        D_concat = action_concat.size(-1)
        D_raw = int(getattr(self, "action_dim_raw", D_concat))
        if D_raw <= 0 or D_concat % D_raw != 0:
            # Fallback: hand back as-is, no buffering. Caller will likely hit a shape error,
            # but at least we don't silently mis-slice. Surface clearly via assert.
            assert D_concat == D_raw, (
                f"BC frameskip mismatch: predictor action_dim={D_concat}, raw action_dim_raw="
                f"{D_raw}. Set self.action_dim_raw to a divisor (frameskip × raw)."
            )
            return action_concat
        n_frames = D_concat // D_raw
        # (E, n_frames, D_raw) → list of n_frames × (E, D_raw); FIFO across env steps.
        per_frame = action_concat.view(E, n_frames, D_raw).transpose(0, 1).contiguous()
        self._bc_buffer = [per_frame[i] for i in range(n_frames)]
        return self._bc_buffer.pop(0)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        info_dict = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in info_dict.items()
        }

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"].unsqueeze(1)  # (B, 1, T, D) for broadcast
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)

        return cost
