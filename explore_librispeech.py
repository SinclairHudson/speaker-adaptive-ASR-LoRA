# This file explores librispeech dataset, finding different statistics for the different speakers, and splits
from datasets import load_dataset
import matplotlib.pyplot as plt

dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean",
                       split="validation", cache_dir="/media/sinclair/M1/huggingface/datasets")

def analyse_speaker_distribution():
    speakers = {}
    for item in dataset:
        breakpoint()
        if item['speaker_id'] in speakers:
            speakers[item['speaker_id']] += 1
        else:
            speakers[item['speaker_id']] = 1

    # sort the speakers by frequency of occurence
    speakers = dict(sorted(speakers.items(), key=lambda item: item[1], reverse=True))
    # TODO this bar chart will be dumb, speakers don't need their numerical ids
    plt.bar(speakers.keys(), speakers.values())
    plt.show()

if "__main__" == __name__:
    analyse_speaker_distribution()
