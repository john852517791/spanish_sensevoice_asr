"""Model, data, and decoding helpers for Spanish ASR."""

__all__ = ["SenseVoiceSmallCTC", "SpanishAsrTokenizer"]


def __getattr__(name: str):
	if name == "SenseVoiceSmallCTC":
		from .ctc_model import SenseVoiceSmallCTC

		return SenseVoiceSmallCTC
	if name == "SpanishAsrTokenizer":
		from .tokenizer import SpanishAsrTokenizer

		return SpanishAsrTokenizer
	raise AttributeError(name)
