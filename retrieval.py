from datasets import load_dataset
import pandas as pd
import os
import json
import re
from pyserini.search.lucene import LuceneSearcher
import subprocess
import sys

# 加载知识库
wikitext = load_dataset("parquet", data_files="data/wikitext/validation-00000-of-00001.parquet", split="train")
wikitext_df = wikitext.to_pandas()

# 加载禁止标题列表
with open('data/wikitext/wikitext103_forbidden_titles.txt', 'r', encoding='utf-8') as f:
    forbidden_titles = set(line.strip() for line in f)

# 加载查询数据
nq_dev_local = load_dataset("parquet", data_files="data/nq/validation-00000-of-00001.parquet", split="train")
nq_retrieval = pd.DataFrame()
nq_retrieval['question'] = nq_dev_local['question']
nq_retrieval['answer'] = nq_dev_local['answer']

# 准备知识库以进行索引
os.makedirs('wikitext_collection/docs', exist_ok=True)

def extract_title(text):
    lines = text.strip().split('\n')
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('=') and stripped_line.endswith('='):
            # 去除开头和结尾的 '=' 以及空格
            title = stripped_line.strip('=').strip()
            return title
    return ''

def write_jsonl(df, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            text = row['text']
            title = extract_title(text)
            contents = f'"{title}"\n{text}'
            doc = {'id': str(idx), 'contents': contents}
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')

write_jsonl(wikitext_df, 'wikitext_collection/docs/wikitext_corpus.jsonl')

# 索引知识库
index_command = [
    sys.executable, '-m', 'pyserini.index.lucene',
    '--collection', 'JsonCollection',
    '--input', 'wikitext_collection',
    '--index', 'indexes/wikitext_index',
    '--generator', 'DefaultLuceneDocumentGenerator',
    '--threads', '1',
    '--storePositions', '--storeDocvectors', '--storeRaw', '--storeContents'  # 添加 '--storeContents'
]
subprocess.run(index_command)

# 初始化检索器
searcher = LuceneSearcher('indexes/wikitext_index')

# 定义清洗函数
def clean_text(text):
    # 去除首尾空白字符
    text = text.strip()
    # 替换换行符为空格
    text = text.replace('\n', ' ')
    # 替换多个空白字符为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 移除非打印字符
    text = ''.join(c for c in text if c.isprintable())
    # 替换多个连续的引号为单个引号
    text = re.sub(r'"+', '"', text)
    return text

# 定义检索函数
def retrieve_docs(searcher, query, forbidden_titles, k=5):
    k_multiplier = 20  # 为了补偿被过滤的文档，检索更多的结果
    hits = searcher.search(query, k=k*k_multiplier)
    results = []
    for hit in hits:
        res_dict = json.loads(hit.raw)
        context_str = res_dict['contents']
        # 提取标题
        title_line = context_str.split('\n', 1)[0]
        if title_line.startswith('"') and title_line.endswith('"'):
            title = title_line[1:-1]
        else:
            title = title_line
        if title in forbidden_titles:
            continue  # 跳过禁止的标题
        # 清洗内容
        cleaned_contents = clean_text(context_str)
        doc = {
            'docid': hit.docid,
            'score': hit.score,
            'contents': cleaned_contents
        }
        results.append(doc)
        if len(results) >= k:
            break
    return results

# 对每个问题执行检索
retrieval_results = []
for query in nq_retrieval['question']:
    results = retrieve_docs(searcher, query, forbidden_titles, k=5)
    retrieval_results.append(results)

nq_retrieval['retrieval_results'] = retrieval_results
# nq_retrieval.to_csv('data/nq_retrieval_results.csv', index=False)
nq_retrieval_records = nq_retrieval.to_dict(orient='records')
with open('data/nq_retrieval_results.json', 'w', encoding='utf-8') as f:
    json.dump(nq_retrieval_records, f, ensure_ascii=False, indent=4)
