import torch
from scipy.io.wavfile import write

# Load Tacotron2 model
tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
tacotron2 = tacotron2.to('cuda')
tacotron2.eval()

# Load WaveGlow model
waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to('cuda')
waveglow.eval()

# Load TTS utils for preprocessing
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')

# Input text
text = "Hello, I am very happy to be your technical assistant!, can you please tell what help should i do for you?"

# Prepare input sequences
sequences, lengths = utils.prepare_input_sequence([text])

# Perform inference
with torch.no_grad():
    mel, _, _ = tacotron2.infer(sequences, lengths)
    audio = waveglow.infer(mel)

# Convert audio to numpy array
audio_numpy = audio[0].data.cpu().numpy()

# Save audio as WAV file
rate = 22050
write("audio.wav", rate, audio_numpy)
