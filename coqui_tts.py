from TTS.api import TTS

def text_to_speech(text, model_name="tts_models/en/blizzard2013/capacitron-t2-c150_v2", file_path="intro.wav", progress_bar=True, gpu=True):
    # Initialize TTS with the target model name
    tts = TTS(model_name=model_name, progress_bar=progress_bar, gpu=gpu)
    # Run TTS and save to file
    tts.tts_to_file(text=text, file_path=file_path)

if __name__ == "__main__":
    text = "Hello guys and welcome back to another video!"
    text_to_speech(text=text, file_path="intro.wav")
