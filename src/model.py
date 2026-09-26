import torch
from transformers import (
	AutoTokenizer,
	Gemma3ForConditionalGeneration
)

MODEL_ID = "google/gemma-3-4b-it"


def load_model():
	tokenizer = AutoTokenizer.from_pretrained(
		MODEL_ID
	)

	model = Gemma3ForConditionalGeneration.from_pretrained(
		MODEL_ID,
		dtype=torch.bfloat16,
		device_map="auto"
	)

	model.eval()

	return tokenizer, model


