import openai
import pandas as pd
from tqdm import tqdm
import json

# 设置 OpenAI API 密钥
openai.api_key = ""

def compress_document(question, document, word_count):
    """
    调用 GPT-3.5 API 压缩文档
    :param question: 问题 q
    :param document: 检索到的文档 d
    :param word_count: 文档总词数
    :return: 压缩后的文档摘要
    """
    # 计算目标摘要字数
    summary_word_count = int(word_count * 0.3)

    # 构建 Prompt
    prompt = f"""
Compress the information in the retrieved documents into a summary that is 30% of the original document's word count. Please note that the word count of the summary must be strictly controlled to {summary_word_count} words.
Retrieved documents: {document}
"""

    # 调用 OpenAI ChatCompletion API
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in document summarization."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,  # 可根据需求调整生成内容的随机性
        )
        # print("API Response:", response)

        # 返回生成的摘要
        return response['choices'][0]['message']['content'].strip()

    except openai.error.OpenAIError as e:
        print(f"Error calling OpenAI API: {e.__class__.__name__}: {e}")
        print(f"Error calling OpenAI API: {e}")
        return None


# 示例使用
if __name__ == "__main__":
    # q = "What is the name of manchester united stadium?"
    # d = """
    # Old Trafford is a football stadium in Old Trafford, Greater Manchester, England, and the home of Manchester United. With a capacity of 74,994, it is the largest club football stadium (and second largest football stadium overall after Wembley Stadium) in the United Kingdom, and the eleventh-largest in Europe. It is about from Old Trafford Cricket Ground and the adjacent tram stop. Nicknamed "The Theatre of Dreams" by Bobby Charlton, Old Trafford has been United's home ground since 1910, although from 1941 to 1949 the club shared Maine Road with local rivals Manchester City as a result of Second.
    # """
    # word_count = len(d.split())
    # compressed_summary = compress_document(q, d, word_count)
    # print("Compressed Summary:", compressed_summary)


    # 读取json文件
    result = []
    nq_retrieval = pd.read_json('inputs/nq-test-dense-results.json')

    # 对于nq_retrieval中的每条数据，取前5个ctxs
    nq_retrieval['ctxs'] = nq_retrieval['ctxs'].apply(lambda x: x[:5])
    nq_extractive_compressor = []

    for _,data in tqdm(nq_retrieval.iterrows(), total=len(nq_retrieval)):
        q = data['question']
        for i in range(len(data['ctxs'])):
            d = data['ctxs'][i]['text']
            word_count = len(d.split())
            compressed_summary = compress_document(q, d, word_count)

            # 将nq_retrieval中的数据转换为压缩后的摘要
            if compressed_summary:
                nq_extractive_compressor.append({
                    "question": q,
                    "answers": data['answers'],
                    "ctxs": {
                        'title':data['ctxs'][i]['title'],
                        'text': compressed_summary,
                        'score': data['ctxs'][i]['score'],
                        'id': data['ctxs'][i]['id'],
                    }
                })
            else:
                print("压缩失败")

    # 保存结果
    with open('outputs/3.mt-nq-abstract-compressor.json', 'w') as f:
        json.dump(nq_extractive_compressor, f, indent=4)




