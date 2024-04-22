## Speaker-Adaptive ASR with LoRA

Automatic Speech Recognition (ASR) with Low-Rank Adapters (LoRA).

This was my final project for CSC2518: Spoken Language Processing, a seminar course at the University of Toronto.

The main idea is to see if pre-trained ASR models benefit from explicit speaker information as a prior.
I was concerned that while these large models like Wav2Vec 2.0 might do well on aggregate on benchmarks, but they could be neglecting certain speakers within the benchmarks.
For example, speakers with rare accents or speech patterns.

As it turns out, the Word Error Rate per speaker does actually vary quite a bit:

However, I couldn't build a system that improved speakers across the board, despite trying some pretty funky ideas.
This could be because these models have sufficient capacity to deal with the heterogeneous nature of this data.
