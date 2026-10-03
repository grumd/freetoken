"""``--vision-on-cpu`` weight routing: the vision stack alone leaves the device.

``_materialize_loaded_weight_state_dict`` is the single place that decides where a loaded
tensor lands, so the tower must be the only key family that honours ``vision_device`` and a
non-primary rank must end up with no vision keys at all (its model has no tower to fill).
"""

from __future__ import annotations

import torch

from freetoken.engine.engine import _materialize_loaded_weight_state_dict


def _stream():
    return [
        ("model.embed_tokens.weight", torch.ones(4, 4, dtype=torch.bfloat16)),
        ("vision_tower.patch_embed.proj.weight", torch.ones(3, 3, 2, 4, 4, dtype=torch.float32)),
        ("vision_tower.merger.linear_fc2.weight", torch.ones(2, 2, dtype=torch.float32)),
        ("lm_head.weight", torch.ones(4, 4, dtype=torch.float32)),
    ]


def _model_state():
    # embed_tokens declares bf16 while the stream carries it; the loader must cast it
    return {"model.embed_tokens.weight": torch.empty(4, 4, dtype=torch.bfloat16)}


def test_vision_keys_follow_vision_device():
    host = torch.device("cpu")
    state = _materialize_loaded_weight_state_dict(
        _model_state(), _stream(), device=host, vision_device=host
    )
    assert set(state) == {
        "model.embed_tokens.weight",
        "vision_tower.patch_embed.proj.weight",
        "vision_tower.merger.linear_fc2.weight",
        "lm_head.weight",
    }
    # the per-key dtype cast still applies next to the host-resident tower
    assert state["model.embed_tokens.weight"].dtype == torch.bfloat16
    assert state["vision_tower.merger.linear_fc2.weight"].dtype == torch.float32


def test_non_primary_rank_drops_the_whole_tower():
    host = torch.device("cpu")
    state = _materialize_loaded_weight_state_dict(
        _model_state(), _stream(), device=host, vision_device=host, keep_vision=False
    )
    assert not [k for k in state if k.startswith("vision_tower.")]
    assert "lm_head.weight" in state


def test_default_keeps_everything_on_the_engine_device():
    host = torch.device("cpu")
    state = _materialize_loaded_weight_state_dict(_model_state(), _stream(), device=host)
    assert len(state) == 4
    assert all(t.device == host for t in state.values())
