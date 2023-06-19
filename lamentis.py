header = "Language-based Artificial-Intelligence Model with Enhanced Natural-language-processing and Text-based Intelligent System"
print(header)

#openAI_api_key = "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"
#huggingface_api_key = "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"

import os
import textwrap
import pyttsx3
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import sys

def run_lamentis():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)

    def speak(text):
        engine.say(text)
        engine.runAndWait()

    print("Initializing...")
    speak("Initializing LAMENTIS")

    # Load the HuggingFaceHub API token from the .env file
    load_dotenv(find_dotenv())
    huggingfacehub_api_token = os.environ["HUGGINGFACEHUB_API_TOKEN"]

    # Load the LLM model from the HuggingFaceHub
    repo_id = "tiiuae/falcon-7b-instruct"
    falcon_llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 500})

    # Create a PromptTemplate and LLMChain
    template = """Your name is LAMENTIS, short for Language-based Artificial-Intelligence Model with Enhanced Natural-language-processing and Text-based Intelligent System. 
    You are an Artificial Intelligence Personal Virtual Assistant created by Gianne P. Bacay.

    Question: {question}"""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=falcon_llm)

    # Run the LLMChain
    response = "System is now online."
    print(response)
    speak(response)
    while True:
        question = input("\ninput: ")
        if "quit" in question:
            break
        response = llm_chain.run(question)
        wrapped_text = textwrap.fill(response, width=200, break_long_words=False, replace_whitespace=False)
        print("LAMENTIS: " + wrapped_text)
        speak(response)

if __name__ == '__main__':
    run_lamentis()


#__________________________python lamentis.py