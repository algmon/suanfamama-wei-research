from openai import OpenAI
client = OpenAI()

audio_file= open("./sample.speech.1.m4a", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1",
  file=audio_file
)
print(transcription.text)
