from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from companion.ml.emotion_model import EmotionModelConfig, EmotionTransformer  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "emotion"


def test_forward_shape():
    model = EmotionTransformer(EmotionModelConfig(vocab_size=50, n_classes=6, d_model=32, n_heads=4,
                                                  n_layers=2, d_ff=64, max_len=16))
    out = model(torch.randint(1, 50, (3, 10)))
    assert out.shape == (3, 6)


def test_padding_does_not_change_prediction():
    torch.manual_seed(0)
    model = EmotionTransformer(EmotionModelConfig(vocab_size=50, n_classes=6, d_model=32, n_heads=4,
                                                  n_layers=2, d_ff=64, max_len=16)).eval()
    ids = torch.tensor([[5, 9, 13, 2]])
    padded = torch.tensor([[5, 9, 13, 2, 0, 0, 0]])
    with torch.no_grad():
        assert torch.allclose(model(ids), model(padded), atol=1e-5)


@pytest.mark.skipif(not (ARTIFACTS / "model.pt").exists(), reason="emotion model not trained")
def test_trained_classifier_on_clear_cases():
    from companion.hooks import EmotionHook
    from companion.ml.emotion import EmotionClassifier

    clf = EmotionClassifier(ARTIFACTS)
    assert clf.predict("my grandma passed away last night")[0] == "sadness"
    assert clf.predict("I'm really nervous about my presentation")[0] == "fear"
    assert clf.predict("open my downloads folder")[0] == "neutral"

    events = []
    context = EmotionHook(clf).before_turn("I feel kind of lonely today", events.append)
    assert events[0].data["label"] == "sadness" and "sadness" in context
