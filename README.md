# NMA_Project_SER

# 1. Run in colab environment
!pip install transformers torchaudio librosa -q
!pip install SpeechRecognition -q
!pip install librosa gensim -q
!pip install tqdm -q
!apt-get install unzip -q
!pip install wandb -q

from google.colab import drive
drive.mount('/content/drive')

import os
!git clone https://github.com/kafkapple/NMA_Project_SER.git
os.chdir('./drive/MyDrive/NMA_Project_SER')
!ls

!python main_hyperparam_search.py

