# %%
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI

load_dotenv(find_dotenv())

# %% img2text
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]["generate_text"]

    print(text)
    return text

img2text("image.jpg")

# %% llmtext
"""
def generate_story(scenario):
    template= 
    You are a stpry teller:
    
    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    story_llm = LLMChain(llm=OpenAI(
        model_name="gpt-3.5-turbo",temperature=1), prompt=prompt, verbose=True)
    story = story_llm.predict(scneario=scenario)

    print(story)
    return story
"""

scenario = img2text("image.jpg")
story = generate_story(scenario)

# %% text2speech