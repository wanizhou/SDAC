import argparse
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn.functional as F
from nltk.corpus import stopwords
import pandas as pd
import nltk
from tqdm import tqdm
import json
import re


# 清理文本中的特殊字符
def clean_text(text):
    """使用正则表达式去除特殊字符"""
    return re.sub(r'[\u2013\u2014\u2026\u201c\u201d]', '', text)


# 计算每个单词的TF-IDF得分并结合稀疏注意力机制
def compute_word_importance_for_doc(retrieval_doc, tokenizer, model, device, stop_words, threshold):
    """计算每个文档中每个词的重要性"""
    inputs = tokenizer(retrieval_doc['text'], return_tensors="pt").to(device)

    # 清理文本中的特殊字符
    cleaned_text = clean_text(retrieval_doc['text'])
    inputs = tokenizer(cleaned_text, return_tensors="pt").to(device)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_dim)

    # Step 1: 使用TF-IDF计算词汇重要性
    tokens = tokenizer.tokenize(retrieval_doc['text'])
    tokenized_text = " ".join(tokens).replace(" ##", "")  # 去掉子词前缀"##"以兼容TF-IDF

    tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda text: text.split())  # 使用空格分词
    tfidf_matrix = tfidf_vectorizer.fit_transform([tokenized_text])
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = dict(zip(tfidf_feature_names, tfidf_matrix.toarray()[0]))

    # Step 2: 稀疏注意力机制
    query = embeddings
    key = embeddings
    value = embeddings

    # 计算注意力分数
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (embeddings.size(-1) ** 0.5)
    attention_probs = F.softmax(attention_scores, dim=-1)

    # 稀疏化注意力矩阵：设定阈值，忽略低于阈值的注意力分数
    sparse_attention_probs = attention_probs * (attention_probs > 0.05).float()
    sparse_attention_probs = (sparse_attention_probs - sparse_attention_probs.min()) / (
                sparse_attention_probs.max() - sparse_attention_probs.min())

    # Step 3: 结合TF-IDF与稀疏注意力
    tfidf_weights = []
    for token_id in inputs['input_ids'].squeeze().cpu():
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        if token.lower() in stop_words:
            tfidf_weights.append(0.1)  # 对停用词给予较低权重
        elif token in ['[CLS]', '[SEP]', '.', ',', '!', '?', '-', ';', ':', '(', ')', '[', ']', '{', '}', '<', '>',
                       '\'', '\"', '\'s', '\' s', '-']:
            tfidf_weights.append(0)  # 忽略标点符号
        else:
            tfidf_weights.append(tfidf_scores.get(token, 0))

    tfidf_weights = torch.tensor(tfidf_weights, device=device).unsqueeze(0)
    tfidf_weights = (tfidf_weights - tfidf_weights.min()) / (tfidf_weights.max() - tfidf_weights.min())

    # 使用加权平均代替乘法计算最终的重要性
    combined_importance = 0.5 * sparse_attention_probs.sum(dim=-1) + 0.5 * tfidf_weights

    # Step 4: 构造词及其重要性字典
    word_importance = {}
    token_occurrences = {}
    for idx, token_id in enumerate(inputs["input_ids"].squeeze()):
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        if tfidf_weights[0, idx].item() != 0:
            token_occurrences[token] = token_occurrences.get(token, 0) + 1
            unique_key = f"{token}_{token_occurrences[token]}"
            word_importance[unique_key] = round(combined_importance[0, idx].item(), 5)

    # Step 5: 筛选低重要性的词
    num_items = int(len(word_importance) * threshold) - 1
    threshold_value = sorted(word_importance.values(), reverse=True)[num_items] if 0 <= num_items < len(
        word_importance) else float('-inf')
    filtered_word_importance = {key: importance for key, importance in word_importance.items() if
                                importance >= threshold_value}

    return filtered_word_importance


# 主计算函数
def compute_word_importance(data, tokenizer, model, device, stop_words, threshold):
    """处理文档数据并计算每个文档的词汇重要性"""
    ctxs = []
    for retrieval_doc in data['ctxs']:
        # 计算每个文档的词汇重要性
        filtered_word_importance = compute_word_importance_for_doc(retrieval_doc, tokenizer, model, device, stop_words,
                                                                   threshold)

        # 构造包含筛选后词汇的结果
        ctxs.append({
            "words_score": filtered_word_importance,
            "title": retrieval_doc['title'],
            "score": retrieval_doc['score'],
            "text": " ".join([key.split('_')[0] for key in filtered_word_importance.keys()])
        })

    results = {'question': data['question'], 'answers': data['answers'], 'ctxs': ctxs}
    return results


def main(args):
    """主程序入口"""
    # 下载nltk停用词表
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    # 加载E5模型和tokenizer到CUDA
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    # 读取json文件
    nq_retrieval = pd.read_json(args.input_file)

    # 对于nq_retrieval中的每条数据，取前5个ctxs
    nq_retrieval['ctxs'] = nq_retrieval['ctxs'].apply(lambda x: x[:args.top_k])

    nq_extractive_compressor = []

    # 计算每条数据的词汇重要性
    for _, data in tqdm(nq_retrieval.iterrows(), total=len(nq_retrieval)):
        results = compute_word_importance(data, tokenizer, model, device, stop_words, threshold=args.threshold)
        nq_extractive_compressor.append(results)

    # 保存为json文件
    with open(args.output_file, "w") as file:
        json.dump(nq_extractive_compressor, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # 使用argparse来解析命令行参数
    parser = argparse.ArgumentParser(description="Semantic Compression of Document Retrieval")

    # 添加命令行参数
    parser.add_argument('--model_name', type=str, default='intfloat/e5-base',
                        help='Model name or path to pre-trained model')
    parser.add_argument('--input_file', type=str, default='inputs/trivia-retrieval-documents.json',
                        help='Input JSON file with retrieval results')
    parser.add_argument('--output_file', type=str, default='outputs/semantic_compressor_results.json',
                        help='Output file to save results')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top ctxs to keep from each retrieval')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Threshold for filtering words based on importance score')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run the model on (cuda or cpu)')

    # 解析参数
    args = parser.parse_args()

    # 运行主程序
    main(args)
