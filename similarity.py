from vertexai.preview.language_models import TextEmbeddingInput, TextEmbeddingModel
import os
from openai import AzureOpenAI
import openai

from sentence_transformers import SentenceTransformer, util
# from sklearn.metrics import jaccard_score
# from sklearn.preprocessing import MultiLabelBinarizer

def get_azure_similarity_score(sentence1, sentence2):
    # Configure Azure OpenAI client
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = 'https://infinitus-dev.openai.azure.com/'
    deployment_name = 'text-embedding-3-large'
    
    if not api_key or not azure_endpoint or not deployment_name:
        raise ValueError("Missing required Azure OpenAI environment variables")
    
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2023-05-15",
        azure_endpoint=azure_endpoint
    )

    response = client.embeddings.create(input=[sentence1, sentence2], model=deployment_name)
    vec1, vec2 = response.data[0].embedding, response.data[1].embedding
    sim = util.cos_sim(vec1, vec2)
    return sim.item()

def get_gemini_similarity_score(sentence1, sentence2):
    """
    Get similarity score using Google's Generative AI (Gemini) embeddings.
    Falls back to sentence transformers if Gemini is not available.
    """
    import google.generativeai as genai
    # Get embeddings for both sentences using Gemini's embedding model
    model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
    embedding_inputs = [TextEmbeddingInput(sentence1, "RETRIEVAL_QUERY"), TextEmbeddingInput(sentence2, "RETRIEVAL_QUERY")]
    response = model.get_embeddings(embedding_inputs)
    
    vec1 = response[0].values
    vec2 = response[1].values
    sim = util.cos_sim(vec1, vec2)
    return sim.item()
        
def is_entity_same(entity: str, value: str, sentence2: str) -> bool:
    prompt = (
        f"Do the two sentences refer to the same {entity}?\n\n"
        f"Value: {value}\n"
        f"Sentence 2: {sentence2}\n"
        f"Answer with 'Yes' or 'No' only."
    )
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = 'https://infinitus-dev.openai.azure.com/'
    deployment_name = 'gpt-4'
    if not api_key or not azure_endpoint or not deployment_name:
        raise ValueError("Missing required Azure OpenAI environment variables")
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2023-05-15",
        azure_endpoint=azure_endpoint
    )
    response = client.chat.completions.create(
        model="gpt-4-2",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3,
        temperature=0,
        n=1,
        stop=None,
    )
    resp = response.choices[0].message.content or ''
    answer = resp.strip()
    if answer == 'No':
        return False
    else:
        return True

# # Example usage:
# entity = "phone number"
# sentence1 = "Call me at 555-123-4567 tomorrow."
# sentence2 = "My number is 5551234567, reach out anytime."
# print(is_entity_same(entity, sentence1, sentence2))  # Expected output: Yes
# sentences = [
#     "I am a student",
#     "I am a teacher",
#     "My zip code is 9 4 1 0 1",
#     "My zip code is 94101",
#     "My zip code is 94105",
#     'My zip code is nine four one zero one',
#     'My zip code is nine four one o one',
#     'My uh zip code huh... it is.. um..  I think nine then four one o one yeah that\'s it',
#     'March tenth, nineteen forty-seven, that was a Monday',
#     "Hi, this is Sarah Johnson. My date of birth is March 15th, 1985."
#     "Yeah, my zip code is 94107. I live in San Francisco."
#     "Sure, my phone number is 312-555-8960."
#     "My name is David Lee. That’s L-E-E."
#     "Can you deliver that on August 4th?"
#     "My email is john.doe92@gmail.com."
#     "I’m calling from 60614, that’s Chicago."
#     "My number is 480-777-4321 in case you need to call me back."
#     "Hi, it's Melissa from 78229. I'm calling about my appointment."
#     "Birthday is December 1st, 1979."
# ]

# pairs = []
# for sentence in sentences:
#     for sentence2 in sentences:
#         pairs.append((sentence, sentence2))

# # randomize the pairs
# import random
# random.shuffle(pairs)

from transformers import T5Tokenizer, T5ForConditionalGeneration
from difflib import SequenceMatcher


model_name = "t5-base"  # Or any public T5 variant
tokenizer = T5Tokenizer.from_pretrained(model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(model_name)


def normalize_with_t5(text: str) -> str:
    input_ids = tokenizer("normalize: " + text, return_tensors="pt").input_ids
    output_ids = t5_model.generate(input_ids, max_new_tokens=32)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def entity_value_similarity(text1: str, text2: str) -> float:
    norm1 = normalize_with_t5(text1)
    norm2 = normalize_with_t5(text2)
    return SequenceMatcher(None, norm1, norm2).ratio()


from sentence_transformers import SentenceTransformer, util

sentence_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def sentence_style_similarity(text1, text2):
    emb1 = sentence_model.encode(normalize_with_t5(text1), convert_to_tensor=True)
    emb2 = sentence_model.encode(normalize_with_t5(text2), convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

import csv
import json

# print for each pair, the sentence and the similarity score in a tabular format
# write as a csv file (tab separated)
offset = 0
with open('dob_similarity.csv', 'w') as w:
    # tab delimiter.
    w.write('value \t additional_value \t type \t sentence1 \t synthetic_type \t sentence2 \t azure_similarity_score \t azure_value_match \t gemini_similarity_score \t gemini_value_match \t match \t generated_is_subset \t generated_is_superset \n') 
    with open('wed_dob_cross.csv', 'r') as f:
        reader = csv.reader(f) # ignore the first row
        next(reader)
        # headers: callee_response,field_extracted_value,types,synthetic_type,synthetic_transcript
        for row in reader:
            if len(row) < 5:
                continue
            empty = False
            if row[0] == 'nan':
                continue
            if row[1] == 'nan':
                continue
            for i in range(len(row)):
                if row[i].strip() == '':
                    empty = True
                    break
            if empty:
                continue
            sentence1 = row[0]
            value = row[1]
            additional_value = row[1+offset]
            types = row[2+offset].replace(' ', '')
            all_types = set(types.split(','))
            synthetic_type = row[3+offset].replace(' ', '')
            all_synthetic_types = set(synthetic_type.split(','))
            sentence2 = row[4+offset]

            s = get_azure_similarity_score(sentence1, sentence2)
            v = 1
            # try:
            #     v = is_entity_same("date of birth", row[1], sentence2)
            # except Exception as e:
            #     print(e)
            #     v = 1
            gs = 0
            gv = 0
            gs = get_gemini_similarity_score(sentence1, sentence2)
            gv = 1
            # try:
            #     gv = is_entity_same("date of birth", row[1], sentence2)
            # except Exception as e:
            #     print(e)
            #     gv = 1
            if v:
                v = 1
            else:
                v = 0
            if gv:
                gv = 1
            else:
                gv = 0


            match = False
            generated_is_subset = False
            generated_is_superset = False

            if all_types.issubset(all_synthetic_types):
                generated_is_superset = True
            if all_synthetic_types.issubset(all_types):
                generated_is_subset = True
            if all_types == all_synthetic_types:
                match = True


            w.write(f'{value} \t {types} \t {sentence1} \t {synthetic_type} \t {sentence2} \t {s} \t {v} \t {gs} \t {gv} \t {match} \t {generated_is_subset} \t {generated_is_superset} \n')
            print("value: ", value)
            print("additional_value: ", additional_value)
            print("types: ", types)
            print(sentence1)
            print(synthetic_type)
            print(sentence2)
            print(s)
            print(v)
            print(gs)
            print(gv)
            print("match: ", match)
            print("generated_is_subset: ", generated_is_subset)
            print("generated_is_superset: ", generated_is_superset)
            print("--------------------------------")



