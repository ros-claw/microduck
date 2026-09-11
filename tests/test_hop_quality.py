"""Physical negative controls for the complete-hop scorer."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_hop_quality import evaluate


def test_standing_is_not_counted_as_hopping():
    result = evaluate(ROOT.parent / "microduck/policies/alpha_stand.onnx", seconds=3)
    assert result["attempts"] == result["clean_hops"] == 0
    assert result["nonfoot_contact_s"] == 0


def test_historical_head_supported_hops_are_rejected():
    result = evaluate(ROOT / "policies/ropehop_classic.onnx", seconds=3)
    assert result["attempts"] > 0
    assert result["nonfoot_contact_s"] > .5
    assert result["clean_hops"] == 0
