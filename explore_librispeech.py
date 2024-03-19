# This file explores librispeech dataset, finding different statistics for the different speakers, and splits
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm

# librispeech is one of: train.clean.100, train.clean.360, train.other.500, validation.clean, validation.other, test.clean, test.other

def get_speakers_and_frequencies(dataset):
    speakers = {}
    for item in tqdm(dataset):
        if item['speaker_id'] in speakers:
            speakers[item['speaker_id']] += 1
        else:
            speakers[item['speaker_id']] = 1
    speakers = dict(sorted(speakers.items(), key=lambda item: item[1], reverse=True))
    return speakers

def analyse_speaker_distribution(dataset):
    speakers = {}
    for item in tqdm(dataset):
        if item['speaker_id'] in speakers:
            speakers[item['speaker_id']] += 1
        else:
            speakers[item['speaker_id']] = 1

    # sort the speakers by frequency of occurence
    speakers = dict(sorted(speakers.items(), key=lambda item: item[1], reverse=True))
    # TODO this bar chart will be dumb, speakers don't need their numerical ids
    breakpoint()
    plt.bar(speakers.keys(), speakers.values())
    plt.show()

def analyse_speaker_overlap_train_test():
    speaker_dicts = []
    for dataset_split in ['train.clean.100', 'train.clean.360', 'train.other.500', 'validation.clean', 'validation.other', 'test.clean', 'test.other']:
        dataset = load_dataset('librispeech_asr', split=dataset_split, cache_dir="/media/sinclair/M1/huggingface/datasets")
        speakers = get_speakers_and_frequencies(dataset)
        print(f"Dataset: {dataset_split}, speakers: {len(speakers)}")
        speaker_dicts.append(speakers)

    for i in range(len(speaker_dicts)):
        for j in range(i+1, len(speaker_dicts)):
            overlap = len(set(speaker_dicts[i].keys()).intersection(set(speaker_dicts[j].keys())))
            print(f"Overlap between {i} and {j}: {overlap}")

if "__main__" == __name__:
    analyse_speaker_overlap_train_test()
