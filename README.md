# Unsupervised Keyphrase Extraction
This is code support chinese keyphrase extraction for EMNLP 2021 paper: [Unsupervised Keyphrase Extraction by Jointly Modeling Local and Global Context](https://aclanthology.org/2021.emnlp-main.14/).

This code support document length > 512.


## requirements
- transformers==4.7.0
- nltk
- pytorch
- tqdm

We employ StanfordCoreNLP 4.5.1 to preprocess the data, you can download it here: https://stanfordnlp.github.io/CoreNLP/index.html.

We employ the chinese bert from https://huggingface.co/hfl/chinese-macbert-base/tree/main.

## Runing
Step 0: tokenize and tag the plain text (one example/line).
```shell
python  src/data_preprocess.py [data_path] [file_name]
```

Step 1: obtain embeddings of candidate phrases and the whole document.
```shell
python src/get_embedding.py --file_path [data_path] --file_name [file_name] --model_name [pretrained model name/path]
```

Step 2: extract keyphrases
```shell
python src/ranker.py [data_path] [model_name]
```

