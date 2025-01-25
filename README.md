# SDAC
This is the repository for the paper "A Semantic-Driven Framework for Adaptive Compression in Document Retrieva".

# 0. Setup
To install the required packages, please run the following command:
```bash
pip install -r requirements.txt
```
To have a Pytorch version specific to your CUDA, [install](https://pytorch.org/) your version before running the above command.
Input test files:https://drive.google.com/drive/folders/1Dg_9pg7K78_wU9JCG1_58KajkIdOU5mT?usp=drive_link

# 1.Data
We have prepared the original retrieval documents of the Natural Questions and TriviaQA datasets in the inputs folder, 
sourced from [here](https://github.com/AI21Labs/in-context-ralm).  
### Dataset Format Instructions

If you need to upload your own dataset, please follow the format below:

```json
[
    {
        "question": "Question text",  // The question field, provide the specific question text
        "answers": [
            "Answer 1",  // The answer field, contains one or more possible answers
            "Answer 2"
        ],
        "ctxs": [  // Context array, each context contains the following fields
            {
                "id": "Context ID",  // Unique identifier for the context
                "title": "Context title",  // The title or name of the context
                "text": "Context content",  // The detailed text of the context, usually a paragraph or description
                "score": "Relevance score",  // (Optional) The relevance score of the context, higher values indicate stronger relevance
                "has_answer": true  // (Optional) Boolean value, indicates whether the context contains an answer
            },
            ...
        ]
    },
    ...
]
```

# 2.Model
#### When performing keyword compression, the _gpt2-xl_ is used by default.  
#### When executing key statement compression, the _e5_ is used by default. It can be replaced as needed.

# 3.Keyword-based compression quick start
To run Keyword-based compression, please use the following command
```bash
python keyword_compression.py 
--input_file  $INPUT_FILE
--output_file $OUTPUT_FILE
--model_name  $MODEL_NAME
--output_file $OUTPUT_FILE
--compression_ratio 0.5
```
# 4.Key statement-based compression quick start
To run Key statement-based compression, please use the following command
```bash
python key_statement_compression.py 
--input_file  $INPUT_FILE
--output_file $OUTPUT_FILE
--model_name  $MODEL_NAME
--output_file $OUTPUT_DIR
--threshold 0.5
```


