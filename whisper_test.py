import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def use_gpt(query):
  client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
  response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {"role": "system", "content": "You are helpfull assistance"},
      {"role": "user", "content": query}
    ],
    temperature=0,
    max_tokens=1000
  )
  return response.choices[0].message.content



client = OpenAI()


audio_file= open("nagranie1.m4a", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1",
  file=audio_file
)
print(transcription.text)


summary = use_gpt("Opowiedz mi o tej osobie na podstawie tego tekstu "+transcription.text)

print(summary)