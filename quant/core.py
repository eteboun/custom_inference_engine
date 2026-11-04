import torch

from attn.core import merge_projections


FP8_MAX = 448
ALL_PROJECTIONS = frozenset({"qkv", "o_proj", "gate_proj", "up_proj", "down_proj"})


def get_scale(x: torch.Tensor) -> torch.Tensor:
    return x.abs().max().to(torch.float32) / FP8_MAX


def quantize(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    x_scaled = (x / scale).clamp(-FP8_MAX, FP8_MAX)
    return x_scaled.to(torch.float8_e4m3fn)


class Fp8Linear(torch.nn.Module):

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
        super().__init__()
        self.fixed = False

        with torch.no_grad():
            scale = get_scale(weight)
            weight_fp8 = quantize(weight, scale)

        self.register_buffer("sw", scale, persistent=False)
        self.register_buffer("w", weight_fp8, persistent=False)
        self.register_buffer(
            "bias",
            None if bias is None else bias.detach(),
            persistent=False
        )
        self.register_buffer(
            "max_inp_s",
            torch.tensor(-float("inf"), dtype=torch.float32, device=weight.device),
            persistent=False
        )

    def observe(self, x: torch.Tensor) -> None:
        if not self.fixed:
            self.max_inp_s.copy_(torch.maximum(self.max_inp_s, get_scale(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.observe(x)

        return torch._scaled_mm(
            quantize(x, self.max_inp_s).view(-1, x.shape[-1]),
            self.w.t(),
            scale_a=self.max_inp_s,
            scale_b=self.sw,
            bias=self.bias,
            out_dtype=torch.float16
        ).view(*x.shape[:-1], -1)


def use_fp8(model, projections: frozenset[str] = ALL_PROJECTIONS) -> None:
    for layer in model.model.layers:
        attention = layer.self_attn

        if "o_proj" in projections:
            attention.o_proj = Fp8Linear(attention.o_proj.weight, attention.o_proj.bias)

        if "qkv" in projections:
            merged = merge_projections([attention.q_proj, attention.k_proj, attention.v_proj])
            qkv_fp8 = Fp8Linear(merged.weight, merged.bias)
            attention.qkv_fp8 = qkv_fp8
            attention.q_proj.register_forward_pre_hook(
                lambda _, args, qkv_fp8=qkv_fp8: qkv_fp8.observe(args[0])
            )

        mlp = layer.mlp
        for name in ("gate_proj", "up_proj", "down_proj"):
            if name in projections:
                linear = getattr(mlp, name)
                setattr(mlp, name, Fp8Linear(linear.weight, linear.bias))


def finalize_fp8_scales(model, margin: float) -> None:
    for module in model.modules():
        if isinstance(module, Fp8Linear):
            if not module.max_inp_s > 0:
                raise RuntimeError("Fp8Linear must see an input before its scale is finalized.")
            module.max_inp_s.mul_(margin)
            module.fixed = True
