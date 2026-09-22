"""A TheLab LoRA adapter merges into a model through jobe.model.apply_adapter, and load() picks it up from the environment."""
import pytest

torch = pytest.importorskip("torch")
thelab_lora = pytest.importorskip("thelab.decisions.lora")
nn = torch.nn

from jobe.model import apply_adapter  # noqa: E402


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(32, 8)
        self.q_proj = nn.Linear(8, 8)
        self.lm_head = nn.Linear(8, 32)

    def forward(self, ids):
        return self.lm_head(torch.tanh(self.q_proj(self.emb(ids))))


def test_apply_adapter_merges_the_trained_delta(tmp_path):
    torch.manual_seed(0)
    trained = Tiny()
    cfg = thelab_lora.LoRAConfig(r=4, alpha=8, dropout=0.0, target_modules=("q_proj",))
    thelab_lora.apply_lora(trained, cfg)
    with torch.no_grad():
        for p in thelab_lora.lora_parameters(trained):
            p.add_(torch.randn_like(p) * 0.2)
    ids = torch.randint(0, 32, (2, 5))
    expected = trained(ids).detach()
    thelab_lora.save_lora(trained, str(tmp_path / "best"), cfg)

    fresh = Tiny()
    fresh.emb.load_state_dict(trained.emb.state_dict())
    fresh.q_proj.load_state_dict(trained.q_proj.base.state_dict())
    fresh.lm_head.load_state_dict(trained.lm_head.state_dict())
    assert not torch.allclose(fresh(ids), expected)                 # base alone differs
    assert apply_adapter(fresh, str(tmp_path / "best")) == 1        # one module merged
    assert torch.allclose(fresh(ids), expected, atol=1e-5)          # merged weights reproduce the trained model
    assert all(not p.requires_grad for p in fresh.parameters())


def test_load_accepts_an_adapter_and_a_missing_one_fails_loudly():
    import inspect
    from jobe.model import load
    assert "lora_dir" in inspect.signature(load).parameters
    with pytest.raises((FileNotFoundError, OSError)):          # never silently skipped
        apply_adapter(Tiny(), "D:/nonexistent/adapter")
