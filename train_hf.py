# from https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md

# !pip install transformers
# !pip install datasets
import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from peft import get_peft_model, LoraConfig

# load pretrained model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


librispeech_samples_ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

# load audio
audio_input, sample_rate = sf.read(librispeech_samples_ds[0]["file"])
print(librispeech_samples_ds[0]["file"])

# pad input values and return pt tensor
input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values


lora_config = LoraConfig(init_lora_weights="gaussian", target_modules=["k_proj", "q_proj", "v_proj", "out_proj"])
peft_model = get_peft_model(model, lora_config)

peft_model_2 = get_peft_model(model, lora_config)

# INFERENCE

# retrieve logits & take argmax
logits = peft_model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)

# transcribe
transcription = processor.decode(predicted_ids[0])

# FINE-TUNE

target_transcription = "A MAN SAID TO THE UNIVERSE I EXIST"

# encode labels
with processor.as_target_processor():
  labels = processor(target_transcription, return_tensors="pt").input_ids

optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-4)
optimizer_2 = torch.optim.AdamW(peft_model_2.parameters(), lr=1e-4)

# this is good, we're only optimizing the lora here now
for name, param in peft_model.named_parameters():
  print(name, param.shape, param.requires_grad)

# compute loss by passing labels
loss = peft_model(input_values, labels=labels).loss
loss.backward()
optimizer.step()
optimizer.zero_grad()

loss = peft_model_2(input_values, labels=labels).loss
loss.backward()
optimizer_2.step()
optimizer_2.zero_grad()
# this shares the base model, so that's epic!!!!

breakpoint()

# Look at model._modules["wav2vec2"].encoder.layers[1].attention.k_proj.weight.grad
