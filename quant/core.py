import torch

FP8_MAX = 448
ALL_PROJECTIONS = frozenset({"qkv", "o_proj", "gate_proj", "up_proj", "down_proj"})


def get_scale(x: torch.Tensor) -> torch.Tensor:
    return x.abs().max().to(torch.float32) / FP8_MAX


def quantize(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    x_scaled = (x / scale).clamp(-FP8_MAX, FP8_MAX)
    return x_scaled.to(torch.float8_e4m3fn)


class Fp8Linear(torch.nn.Module):

    def __init__(
        self, weight: torch.Tensor, bias: torch.Tensor | None, *,
        track_input_scale: bool = True,
    ) -> None:
        super().__init__()
        self.fixed = False
        self.track_input_scale = track_input_scale

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
            torch.tensor(-float("inf"), dtype=torch.float32, device=weight.device)
            if track_input_scale else None,
            persistent=False
        )

    def observe(self, x: torch.Tensor) -> None:
        if self.track_input_scale and not self.fixed:
            self.max_inp_s.copy_(torch.maximum(self.max_inp_s, get_scale(x)))

    def quantize_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.max_inp_s is None:
            raise RuntimeError("This projection requires a shared FP8 input scale.")
        self.observe(x)
        return quantize(x, self.max_inp_s), self.max_inp_s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_quantized(*self.quantize_input(x))

    def forward_quantized(self, x: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
        return torch._scaled_mm(
            x.view(-1, x.shape[-1]),
            self.w.t(),
            scale_a=input_scale,
            scale_b=self.sw,
            bias=self.bias,
            out_dtype=torch.float16
        ).view(*x.shape[:-1], -1)


def use_fp8(model, projections: frozenset[str] = ALL_PROJECTIONS) -> None:
    from attn.core import wrap_qwen2_attention

    if "qkv" in projections:
        wrap_qwen2_attention(model)

    for layer in model.model.layers:
        attention = layer.self_attn
        if "o_proj" in projections:
            attention.o_proj = Fp8Linear(attention.o_proj.weight, attention.o_proj.bias)

        if "qkv" in projections:
            names = ("qkv",) if hasattr(attention, "qkv") else ("q_proj", "k_proj", "v_proj")
            for name in names:
                linear = getattr(attention, name)
                setattr(attention, name, Fp8Linear(
                    linear.weight, linear.bias,
                    track_input_scale=name not in ("k_proj", "v_proj"),
                ))

        mlp = layer.mlp
        for name in ("gate_proj", "up_proj", "down_proj"):
            if name in projections:
                linear = getattr(mlp, name)
                setattr(mlp, name, Fp8Linear(linear.weight, linear.bias))


def finalize_fp8_scales(model, margin: float) -> None:
    for module in model.modules():
        if isinstance(module, Fp8Linear) and module.track_input_scale:
            if not module.max_inp_s > 0:
                raise RuntimeError("Fp8Linear must see an input before its scale is finalized.")
            module.max_inp_s.mul_(margin)
            module.fixed = True
