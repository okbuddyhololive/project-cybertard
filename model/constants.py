from dataclasses import dataclass
import optax


@dataclass
class InferConfig:
    length: int = 256
    top_p: float = 0.8
    temperature: float = 1.0


@dataclass
class ModelParams:
    layers: int = 28
    d_model: int = 4096
    n_heads: int = 16
    n_vocab: int = 50400
    norm: str = "layernorm"
    pe: str = "rotary"
    pe_rotary_dims: int = 64
    seq: int = 2048
    cores_per_replica: int = 8
    per_replica_batch: int = 1

    # batch size of 2 needs 200gb, 1 needs <16. wtf
    optimizer: optax.chain = optax.chain(
        optax.adaptive_grad_clip(0.001),
        optax.centralize(),
        optax.scale_by_adam(0.99, 0.999),
        optax.additive_weight_decay(1e-3),
        optax.scale(-1e-5),
    )
    sampler = None
