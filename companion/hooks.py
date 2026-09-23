from typing import Callable, Protocol

from .agent_events import Event

Emit = Callable[[Event], None]


class TurnHook(Protocol):
    """Something that looks at each user turn. before_turn may return extra context for the model."""

    def before_turn(self, user_text: str, emit: Emit) -> str | None: ...

    def after_turn(self, user_text: str, answer: str) -> None: ...


class EmotionHook:
    def __init__(self, classifier, min_confidence: float = 0.55):
        self.classifier = classifier
        self.min_confidence = min_confidence

    def before_turn(self, user_text: str, emit: Emit) -> str | None:
        label, conf, probs = self.classifier.predict(user_text)
        # A shaky guess shows as neutral rather than a wrong face.
        shown = label if conf >= self.min_confidence else "neutral"
        emit(Event("emotion", {"label": shown, "raw_label": label, "confidence": conf, "probs": probs}))
        if shown == "neutral":
            return None
        return (f"The user's last message sounds {shown} (emotion model confidence {conf:.2f}). "
                "Let that shape your tone, but don't mention the detection.")

    def after_turn(self, user_text: str, answer: str) -> None:
        pass
