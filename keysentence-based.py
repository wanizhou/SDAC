import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm
import json
import spacy
import math


# 文档切分
def split_text(text, nlp):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences


# 计算文档的语义丰富度
def calculate_semantic_richness(model, tokenizer, retrieval_results, max_length, device):
    semantic_richness_scores = []
    for sentences in retrieval_results:
        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=max_length).to(
            device)
        outputs = model(**inputs)

        token_embeddings = outputs[0].masked_fill(~inputs['attention_mask'][..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / inputs['attention_mask'].sum(dim=1)[..., None]
        embedding_variance = sentence_embeddings.var(dim=0).mean().item()

        document_embedding = sentence_embeddings.mean(dim=0)
        cosine_similarities = [
            torch.cosine_similarity(document_embedding, sentence_embeddings[i], dim=0).item()
            for i in range(len(sentences))
        ]
        avg_cosine_similarity = np.mean(cosine_similarities)

        # 检查是否为 NaN 值并处理
        if math.isnan(embedding_variance) or math.isnan(avg_cosine_similarity):
            semantic_richness_score = 0  # 或者选择一个合理的默认值
        else:
            semantic_richness_score = embedding_variance + avg_cosine_similarity

        semantic_richness_scores.append(semantic_richness_score)
    return semantic_richness_scores


def determine_dynamic_window_size(semantic_richness_score, scaling_factor, min_window):
    # 处理语义丰富度得分为 0 的情况，避免除以零
    if semantic_richness_score == 0:
        return min_window  # 使用最小窗口大小

    window_size = max(min_window, int(scaling_factor / semantic_richness_score))
    return window_size


# 根据窗口大小分割文本
def get_text_windows(sentences, window_size):
    windows = []
    for i in range(0, len(sentences), window_size):
        window = sentences[i:i + window_size]
        windows.append(window)  # 直接保留句子列表，而不是将句子拼接为字符串
    return windows


def get_contriever_scores(model, tokenizer, data_row, config):
    retrieval_results = [x['text'] for x in data_row['ctxs']]

    # 对于每个检索文档，进行文本切分，形成一个5*sentences的二维数组
    nlp = spacy.load(config['spacy_model'])
    retrieval_results = [split_text(x, nlp) for x in retrieval_results]
    retrieval_senteces_len = retrieval_results

    # 计算文档的语义丰富度
    semantic_richness_scores = calculate_semantic_richness(model, tokenizer, retrieval_results,
                                                           config['tokenizer_max_length'], config['device'])

    # 根据文档的语义丰富度，动态计算每个文档所需的句子滑动窗口大小
    window_sizes = [determine_dynamic_window_size(x, config['scaling_factor'], config['min_window']) for x in
                    semantic_richness_scores]

    # 使用每个文档的窗口大小过滤掉每个文档中多余的句子
    retrieval_windows = [get_text_windows(x, y) for x, y in zip(retrieval_results, window_sizes)]

    # print(
    #     f"文档数:{len(retrieval_windows)}，每个文档中原来句子的长度:{[len(x) for x in retrieval_results]}，每个文档窗口大小:{window_sizes}")

    # 计算问题的嵌入
    question = data_row['question']
    question_inputs = tokenizer(question, return_tensors='pt', max_length=config['tokenizer_max_length'],
                                truncation=True).to(config['device'])
    question_embedding = model(**question_inputs)[0].mean(dim=1).squeeze()

    # 为每个文档选择最相关的窗口
    best_windows = []
    for doc_index, windows in enumerate(retrieval_windows):
        window_embeddings = []
        for window in windows:
            # 将每个窗口的句子编码为嵌入向量
            inputs = tokenizer(window, padding=True, truncation=True, return_tensors='pt',
                               max_length=config['tokenizer_max_length']).to(config['device'])
            outputs = model(**inputs)[0]
            window_embedding = outputs.mean(dim=1).mean(dim=0)
            window_embeddings.append(window_embedding)

        # 计算每个窗口与问题的余弦相似度
        similarities = [torch.cosine_similarity(question_embedding, emb, dim=0).item() for emb in window_embeddings]
        most_relevant_window = windows[np.argmax(similarities)]
        best_windows.append({
            'title': data_row['ctxs'][doc_index]['title'],
            'text': "".join(most_relevant_window),
            'similarity_score': max(similarities),
            'score': data_row['ctxs'][doc_index]['score'],
            'id': data_row['ctxs'][doc_index]['id'],
        })

    # 生成最终输出格式
    results = {
        "question": data_row['question'],
        "answers": data_row['answers'],
        "ctxs": best_windows
    }
    return results


def main(args):
    # 配置参数
    config = {
        'model_path': args.model_name,
        'tokenizer_max_length': args.tokenizer_max_length,
        'top_k': args.top_k,
        'scaling_factor': args.scaling_factor,
        'min_window': args.min_window,
        'device': args.device,
        'spacy_model': args.spacy_model,
        'input_file': args.input_file,
        'output_file': args.output_file
    }

    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(config['model_path']).to(config['device'])

    # 读取json文件
    nq_retrieval = pd.read_json(config['input_file'])

    # 对于nq_retrieval中的每条数据，取前5个ctxs
    nq_retrieval['ctxs'] = nq_retrieval['ctxs'].apply(lambda x: x[:config['top_k']])
    nq_extractive_compressor = []

    for data in tqdm(nq_retrieval.iterrows(), total=len(nq_retrieval)):
        results = get_contriever_scores(model, tokenizer, data[1], config)
        nq_extractive_compressor.append(results)

    # 保存为json文件
    with open(config['output_file'], "w") as file:
        json.dump(nq_extractive_compressor, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extractive Document Compression")

    # 配置命令行参数
    parser.add_argument('--model_name', type=str, default='gpt2-xl', help='模型路径')
    parser.add_argument('--tokenizer_max_length', type=int, default=512, help='最大token长度')
    parser.add_argument('--top_k', type=int, default=5, help='选择前k个ctxs')
    parser.add_argument('--scaling_factor', type=int, default=3, help='语义丰富度计算的缩放因子')
    parser.add_argument('--min_window', type=int, default=2, help='最小窗口大小')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备配置 (GPU或CPU)')
    parser.add_argument('--spacy_model', type=str, default='en_core_web_sm', help='Spacy模型')
    parser.add_argument('--input_file', type=str, default='inputs/nq-retrieval-documents.json', help='输入文件路径')
    parser.add_argument('--output_file', type=str, default='outputs/1.my-nq-extractive-compressor-results.json',
                        help='输出文件路径')

    args = parser.parse_args()

    # 运行主函数
    main(args)
