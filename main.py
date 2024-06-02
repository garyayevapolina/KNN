import os

from flask import Flask, request, jsonify
import pandas as pd
import requests
import numpy as np
from langchain_core.messages import AIMessage
from langchain.prompts import PromptTemplate
from yandex_chain import YandexLLM
from dotenv import load_dotenv

load_dotenv()


FOLDER_ID = os.environ.get('FOLDER_ID')
IAM_TOKEN = os.environ.get('IAM_TOKEN')

doc_uri = f"emb://{FOLDER_ID}/text-search-doc/latest"
query_uri = f"emb://{FOLDER_ID}/text-search-query/latest"

embed_url = "https://llm.api.cloud.yandex.net:443/foundationModels/v1/textEmbedding"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Api-Key {IAM_TOKEN}",
    "x-folder-id": f"{FOLDER_ID}"
}

prompt_template = """Действуйте как Q&A-бот — Тинькоф Бизнес - Помощь.
Используйте следующий текст в тройных кавычках, чтобы кратко ответить на вопрос пользователя.
Оставьте ссылки, адреса и телефоны как есть. Если ответа в тексте нет, напишите "ответ не найден".
Предоставьте краткий, точный и полезный ответ, иначе вас заменят на GPT-5.

\"\"\"
{context}
\"\"\"

Вопрос студента: {question}"""

prompt = PromptTemplate.from_template(prompt_template)
yagpt = YandexLLM(folder_id="b1g72uajlds114mlufqi",
                  api_key=str(IAM_TOKEN),
                  use_lite=False)
yagpt_chain = prompt | yagpt

app = Flask(__name__)


def get_embedding(text: str, text_type: str = "doc") -> np.array:
    query_data = {
        "modelUri": doc_uri if text_type == "doc" else query_uri,
        "text": text,
    }

    try:
        result = requests.post(embed_url, json=query_data,
                               headers=headers).json()
        print(result)
        return np.array(result["embedding"])
    except:
        print('ERROR', text[:100])
        return []


def get_similar_vec(question_emb):
    df = pd.read_csv('url_with_emb.csv')
    df['embs'] = df['embs'].apply(eval)
    cosine_similarities = df['embs'].apply(
        lambda x: np.dot(x, list(question_emb)))
    nearest_index = np.argmax(cosine_similarities)
    return df['context'][nearest_index], df['url'][nearest_index]


def get_answer(chain, question, context):
    query = {"context": context, "question": question}
    answer = chain.invoke(query)
    if isinstance(answer, AIMessage):
        answer = answer.content
    return answer.strip()


@app.route('/assist', methods=['POST'])
def assist():
    if not request.is_json:
        return jsonify({
            "detail": [{
                "loc": ["body"],
                "msg": "Invalid JSON",
                "type": "type_error"
            }]
        }), 422

    data = request.get_json()

    if 'query' not in data:
        return jsonify({
            "detail": [{
                "loc": ["body", "query"],
                "msg": "Field required",
                "type": "value_error.missing"
            }]
        }), 422

    question_emb = get_embedding(data['query'])
    context, url = get_similar_vec(question_emb)
    answer = get_answer(yagpt_chain, data['query'], context)

    response = {"text": f"Processed query: {answer}", "links": url}
    return jsonify(response), 200


if __name__ == '__main__':
    app.run(debug=True)
