import torch
import torchaudio
import os
from time import time
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice
from tortoise.utils.text import split_and_recombine_text

def generate_speech_from_text(input_text, custom_voice_name="martin", outpath="results/longform/"):
    # Initialize the TextToSpeech model
    tts = TextToSpeech()

    # Process text
    if '|' in input_text:
        print("Found the '|' character in your text, which I will use as a cue for where to split it up. If this was not"
              "your intent, please remove all '|' characters from the input.")
        texts = input_text.split('|')
    else:
        texts = split_and_recombine_text(input_text)

    # Set seed for deterministic results
    seed = int(time())

    # Create output directory
    voice_outpath = os.path.join(outpath, custom_voice_name)
    os.makedirs(voice_outpath, exist_ok=True)

    # Load custom voice samples and conditioning latents
    voice_samples, conditioning_latents = load_voice(custom_voice_name)

    # Generate speech for each part of the text
    all_parts = []
    for j, text in enumerate(texts):
        gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                  preset="fast", k=1, use_deterministic_seed=seed)
        gen = gen.squeeze(0).cpu()
        torchaudio.save(os.path.join(voice_outpath, f'{j}.wav'), gen, 24000)
        all_parts.append(gen)

    # Combine all parts into a single audio file
    full_audio = torch.cat(all_parts, dim=-1)
    combined_audio_path = os.path.join(voice_outpath, 'combined.wav')
    torchaudio.save(combined_audio_path, full_audio, 24000)

    print(f"Audio saved at: {combined_audio_path}")

if __name__ == "__main__":
    input_text = """Your text goes here. This is an example sentence. |
                    You can add multiple sentences separated by the '|' character."""

    generate_speech_from_text(input_text)
