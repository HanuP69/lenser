"""LectureLens Canonical 3-Agent Architecture: AudioAgent, ScreenAgent, SynthesisAgent."""

from lecturelens.agents.base_agent import BaseAgent
from lecturelens.agents.audio_agent import AudioAgent
from lecturelens.agents.screen_agent import ScreenAgent
from lecturelens.agents.synthesis_agent import SynthesisAgent
from lecturelens.agents.structuring_agent import StructuringAgent

__all__ = [
    "BaseAgent",
    "AudioAgent",
    "ScreenAgent",
    "SynthesisAgent",
    "StructuringAgent",
]
