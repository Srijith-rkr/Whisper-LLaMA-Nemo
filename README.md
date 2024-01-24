## Introduction 

This repo is to generate multiple hypotheses for the task of generative error correction. We use a Whisper model in the backend with random temperature sampling to generate diverse candidates, as explained in the paper [Whispering LLaMA: A Cross-Modal Generative Error Correction Framework for Speech Recognition](https://aclanthology.org/2023.emnlp-main.618). You can also find the datasets used in the paper under the data directory. 


## Setup 
The codebase is expected to be compatible with Python 3.8-3.11 and recent PyTorch versions.

Clone the repo 


```bash
git clone https://github.com/Srijith-rkr/Generate_Whisper_hypothesis.git
cd Whisper-LLaMA-Nemo
```
And install its Python dependencies. 

```bash
!pip install -r requirements.txt 
```

## Generating Hypothesis

You can use this [Colab](https://colab.research.google.com/drive/1ZRkbV_hUN-h2RzI53lZ6iqdmjYK8yqFa?usp=sharing) notebook to generate your custom hypothesis or use this simple code snippet.

```bash

  import generate_whisper_hypothesis as whisper 

  # To load your model
  model, _ = whisper.load_model("tiny") # you can also change the whisper model size here. Example model,_ = whisper.load_model("large")

  # To load the audio 
  audio = whisper.load_audio(path)

  # To pad or trim the audio to make it into a 30s input
  audio = whisper.pad_or_trim(audio)

  # To convert the audion into a log mel spectrogram
  mel = whisper.log_mel_spectrogram(audio).to(model.device) 

  # We choose a random temperature for sampling ([70,80] range works good)
  random_temprature = numpy.random.randint(70,81)/100

  # Set fp16 to False if you are using a CPU instance, and you can set the number of candidates you want to generate in the 'best_of' argument
  options = whisper.DecodingOptions(fp16 = False, without_timestamps = True, temperature=random_temprature, best_of = 100)

  result, _ = whisper.decode(model, mel, options)

  return list(result)
```

## Acknowledgement 
This implementation builds on the [Whisper](https://github.com/openai/whisper) repo from OpenAI with only the /whisper/decodeing.py file modified to return multiple candidates. 
