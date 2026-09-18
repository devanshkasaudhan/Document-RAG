import streamlit as st

def generate_with_openai(question, context, key):
    from openai import OpenAI
    client = OpenAI(api_key=key)
    prompt = ("Answer the question using ONLY the context below. "
              "If the answer isn't in the context, say you don't know.\n\n"
              f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2)
    return r.choices[0].message.content.strip()


def generate_with_gemini(question, context, key):
    from google import genai
    client = genai.Client(api_key=key)
    prompt = ("Answer the question using ONLY the context below. "
              "If the answer isn't in the context, say you don't know.\n\n"
              f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt)
    return resp.text.strip()


@st.cache_resource
def get_local_generator():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    return tokenizer, model


def generate_with_local(question, context):
    tokenizer, model = get_local_generator()
    prompt = (f"Answer the question using the context.\n"
              f"Context: {context}\nQuestion: {question}\nAnswer:")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_new_tokens=220)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
