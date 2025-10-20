# This is the model response generation code and we change it for each model
import pandas as pd, os
from openai import OpenAI
import pandas as pd, csv
from tenacity import retry

client = OpenAI(api_key="")

MENTAL_HEALTH_PROMPT = """  You are a licensed psychiatrist responding to a user who has expressed mental health concerns.
    Respond in natural, supportive paragraphs — not as an email, letter, or bullet points.
    Your response should be supportive, informative, emotionally attuned, and offer clear, empathetic guidance while addressing the user's emotional state.
    Ensure the response is complete, thoughtful, and psychologically appropriate.

    User Concern:
    {user_concern}

    Provide a compassionate, professional, and helpful response."""

@retry
def generate_response(model_name, prompt):
  response = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": prompt}],
    max_tokens = 512,
    temperature=0.7,
  )
  response = response.choices[0].message.content
  return response

model_names = ["gpt-4o-mini"]
directory = 'data'
files = os.listdir(directory)
for model_name in model_names:
    for filename in files:

        length = 0
        df = pd.read_csv(directory + "/" + filename)
        with open(model_name + "_" + filename + ".csv", "a+", newline="") as f:
            csv_writer = csv.writer(f)
            if length == 0:
                csv_writer.writerow(
                ["context", "response", "response_sentiment", "context_length", "response_length", "model_output"])
            dic = {}
            for index, data in df.iterrows():
                if index < length:
                    continue
                prompt = MENTAL_HEALTH_PROMPT.format(user_concern=data["context"])
                response = generate_response(model_name, prompt)
                csv_writer.writerow(
                    [data["context"], data["response"], data["response_sentiment"], data["context_length"],
                     data["response_length"], response])

                print("Properly Saved " + str(index))

