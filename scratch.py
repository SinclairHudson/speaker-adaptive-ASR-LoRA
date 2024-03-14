from datasets import load_dataset, load_metric

timit = load_dataset("timit_asr")

print(timit)

# keep speaker_id column
timit = timit.remove_columns(["phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type"])

# TODO eventually, we would like to expose the transformers.wav2vec2Model class, to add a LoRA
# right now though, we're going to be focused on Wav2Vect2ForCTC and try to get the model out for that...

