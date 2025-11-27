from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import copy
import torch.nn.functional as F


class MedTent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=True):  #에피소딕 모드 변경
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

# @torch.enable_grad() # ensure grads in possible no grad context for testing 
# def forward_and_adapt(x, model, optimizer): 
#   """Forward and adapt model on batch of data. 
#   Measure entropy of the model prediction, take gradients, and update params. 
#   """ 
#   # forward 
#   outputs = model(x) 
#   logits = outputs.logits 
#   eps = 1e-8 
#   # binary classification case 
#   if logits.size(-1) == 1: 
#     p = torch.sigmoid(logits) 
#     loss = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps)).mean() 
#   else: # multiclass (softmax) 
#     probs = torch.softmax(logits, dim=-1) 
#     loss = -(probs * torch.log(probs + eps)).sum(dim=-1).mean() 
#   # adapt 
#   # loss = softmax_entropy(outputs).mean(0) 
#   optimizer.zero_grad() 
#   loss.backward() 
#   optimizer.step()

#   return outputs

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(
    x, model, optimizer,
    T=1.3,           # temperature
    lam=0.5,         # diversity 가중치 (IM)
    clip_norm=1.0,   # grad clipping
    #Safe-TTA 게이트
    ent_band=(0.15, 0.60),  # 엔트로피 밴드 (binary)
    div_floor=None,         # 다양성 하한 (None이면 비활성) /multiclass는 자동 정규화
    min_delta=1e-4,         # 엔트로피 최소 개선폭
    eps=1e-8
):
    """
       - 엔트로피 밴드 밖이면 스킵
       - 다양성 낮으면 스킵
       - 업데이트 후 개선이 미미하면 롤백
    """
    #forward (dict 입력이면 언팩해서 호출)
    outputs = model(**x) if isinstance(x, dict) else model(x)
    logits  = outputs.logits
    B, C = logits.shape[0], logits.shape[-1]

    #엔트로피/다양성 계산
    if C == 1:
        # Binary
        p = torch.sigmoid(logits / T).squeeze(-1)        # [B]
        ent_per = -(p*torch.log(p+eps) + (1-p)*torch.log(1-p+eps))  # [B]
        ent_before = ent_per.mean()                      # scalar
        p_bar = p.mean()                                 # scalar
        div = -(p_bar*torch.log(p_bar+eps) + (1-p_bar)*torch.log(1-p_bar+eps))  # scalar
        # Safe gate: entropy band (mean)
        if not (ent_band[0] <= ent_before.item() <= ent_band[1]):
            return outputs  # 스킵
        # Safe gate: diversity floor (옵션)
        if (div_floor is not None) and (div.item() < div_floor):
            return outputs  # 스킵
        loss = ent_before + lam * div
    else:
        # Multiclass
        probs = F.softmax(logits / T, dim=1)             # [B,C]
        ent_per = -(probs * torch.log(probs+eps)).sum(1) # [B]
        ent_before = ent_per.mean()                      # scalar
        p_bar = probs.mean(0)                            # [C]
        div = -(p_bar * torch.log(p_bar+eps)).sum()      # scalar
        # Safe gate: entropy band (정규화하여 밴드 적용)
        H = ent_before.item()
        H_max = torch.log(torch.tensor(C, dtype=probs.dtype, device=probs.device)).item()
        H_norm = H / (H_max + 1e-12)
        if not (ent_band[0] <= H_norm <= ent_band[1]):
            return outputs  # 스킵
        # Safe gate: diversity floor (정규화)
        D = div.item()
        D_norm = D / (H_max + 1e-12)
        if (div_floor is not None) and (D_norm < div_floor):
            return outputs  # 스킵
        loss = ent_before + lam * div

    #백업(롤백 대비)
    params = [p for g in optimizer.param_groups for p in g['params']]
    backup = [p.data.clone() for p in params]
    opt_state = copy.deepcopy(optimizer.state)

    optimizer.zero_grad()
    loss.backward()
    if clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(params, clip_norm)
    optimizer.step()

    #엔트로피 개선폭 검사해서 미미하면 롤백
    with torch.no_grad():
        out2 = model(**x) if isinstance(x, dict) else model(x)
        logits2 = out2.logits
        if C == 1:
            p2 = torch.sigmoid(logits2 / T).squeeze(-1)
            ent_after = -(p2*torch.log(p2+eps) + (1-p2)*torch.log(1-p2+eps)).mean()
        else:
            probs2 = F.softmax(logits2 / T, dim=1)
            ent_after = -(probs2 * torch.log(probs2+eps)).sum(1).mean()

        if (ent_before - ent_after).item() < min_delta:
            # rollback
            for p, b in zip(params, backup):
                p.data.copy_(b)
            optimizer.load_state_dict(opt_state)
            return outputs  #원래 출력 반환

    return out2  #업데이트된 출력 반환



# def collect_params(model):
#     """Collect the affine scale + shift parameters from batch norms.

#     Walk the model's modules and collect all batch normalization parameters.
#     Return the parameters and their names.

#     Note: other choices of parameterization are possible!
#     """
#     params = []
#     names = []
#     for nm, m in model.named_modules():
#         if isinstance(m, nn.BatchNorm2d):
#             for np, p in m.named_parameters():
#                 if np in ['weight', 'bias']:  # weight is scale, bias is shift
#                     params.append(p)
#                     names.append(f"{nm}.{np}")
#     return params, names

def collect_params(model, top_k=4, include_classifier_bias=False):

    for p in model.parameters():
      p.requires_grad_(False)
    
    L_total = model.config.num_hidden_layers
    start = max(0,L_total - top_k)
    
    params = []
    names = []

    for idx, block in enumerate(model.bert.encoder.layer):
        if idx < start:
            continue
        for subname, submod in block.named_modules():
            if isinstance(submod, nn.LayerNorm) and getattr(submod, "bias", None) is not None:
                submod.bias.requires_grad_(True)
                params.append(submod.bias)
                names.append(f"encoder.layer.{idx}.{subname}.bias")

    return params, names



def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


# def configure_model(model):
#     """Configure model for use with tent."""
#     # train mode, because tent optimizes the model to minimize entropy
#     model.train()
#     # disable grad, to (re-)enable only what tent updates
#     model.requires_grad_(False)
#     # configure norm for tent updates: enable grad + force batch statisics
#     for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             m.requires_grad_(True)
#             # force use of batch stats in train and eval modes
#             m.track_running_stats = False
#             m.running_mean = None
#             m.running_var = None
#     return model

def configure_model(model):
    """Configure model for use with tent (LayerNorm version)."""
    model.train()
    model.requires_grad_(False)   # 전체 freeze

    # LayerNorm만 학습 가능하게
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.requires_grad_(True)
            for p in m.parameters():
                p.requires_grad = True
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
