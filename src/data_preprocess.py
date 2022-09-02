from utils import *
import json
from corenlp import StanfordCoreNLP
import nltk
import sys
import os
from tqdm.auto import tqdm

def tokenize_and_tag(nlp, props, sentence):
    json_data = nlp.annotate(sentence, properties=props)
    if type(json_data) is str:
        json_data = json.loads(json_data)
    sentences = []
    sentences_pos_tags = []
    sentences_ner_tags = []
    for sen in json_data['sentences']:
        tokens = []
        pos_tags = []
        ner_tags = []
        for token in sen['tokens']:
            tokens.append(token['word']) 
            pos_tags.append(token['pos'])
            ner_tags.append(token['ner'])
        sentences.append(tokens)
        sentences_pos_tags.append(pos_tags)
        sentences_ner_tags.append(ner_tags)
    return {
        "tokens": sentences,
        "tokens_pos": sentences_pos_tags,
        "tokens_ner": sentences_ner_tags
    }



jar_path = r'/mnt/bn/workspace-lxn/nlp-tools/stanford-corenlp-4.5.1'

lang = 'zh'
# load jar from stanford corenlp path
# memory default is '4g', we set it as '8g'
nlp = StanfordCoreNLP(jar_path, 
                    memory='8g',
                    lang=lang,
                    quiet=True) # if you want to debug, you need to set quiet=False
# Details of annotators: https://stanfordnlp.github.io/CoreNLP/annotators.html
props={
    'annotators': 'tokenize,ssplit,pos,ner', # tokenize, split sentence, part of speech
    'pipelineLanguage': lang, # english
    'outputFormat': 'json' # xml
    }

data_path = sys.argv[1]
data_file = sys.argv[2]

zh_text = load_txt(os.path.join(data_path, data_file))
zh_text = [x.strip().replace("\n", "") for x in zh_text]

res = []
for text in tqdm(zh_text):
    res.append(tokenize_and_tag(nlp, props, text))

nlp.close()
save_jsonline(res, os.path.join(data_path, "test.json"))


# def extract_candidates(tokens_tagged, GRAMMAR, no_subset=False):
#     """
#     Based on part of speech return a list of candidate phrases
#     :param no_subset: if true won't put a candidate which is the subset of an other candidate
#     :return keyphrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
#     """
    

    
#     np_parser = nltk.RegexpParser(GRAMMAR)  # Noun phrase parser
#     keyphrase_candidate = []
#     np_pos_tag_tokens = np_parser.parse(tokens_tagged)
#     count = 0
#     for token in np_pos_tag_tokens:
#         if (isinstance(token, nltk.tree.Tree) and token._label == "NP"):
#             np = ' '.join(word for word, tag in token.leaves())
#             length = len(token.leaves())
#             start_end = (count, count + length)
#             count += length
#             keyphrase_candidate.append((np, start_end))

#         else:
#             count += 1

#     return keyphrase_candidate

# def flat_list(l):
#     return [x for ll in l for x in ll]

# from nltk.corpus import stopwords
# stopword_dict = set(stopwords.words('chinese'))

# data = load_jsonline("data/demo/test.json")
# sentences = flat_list(data[0]['tokens'])
# sentences_pos = flat_list(data[0]['tokens_pos'])

# print(len(sentences))
# GRAMMAR1 = """  NP:
#         {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

# GRAMMAR2 = """  NP:
#         {<JJ|VBG>*<NN.*>{0,3}}  # Adjective(s)(optional) + Noun(s)"""

# GRAMMAR3 = """  NP:
#         {<NN.*|JJ|VBG|VBN>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

# tokens_tagged = []

# tokens_tagged = list(zip(sentences, sentences_pos))
# for i, token in enumerate(sentences):
#     if token.lower() in stopword_dict:
#         tokens_tagged[i] = (token, "IN")

# cands1 = extract_candidates(tokens_tagged, GRAMMAR1)
# cands1 = [x[0].replace(" ", "") for x in cands1]

# cands2 = extract_candidates(tokens_tagged, GRAMMAR2)
# cands2 = [x[0].replace(" ", "") for x in cands2]

# cands3 = extract_candidates(tokens_tagged, GRAMMAR3)
# cands3 = [x[0].replace(" ", "") for x in cands3]

# # print(len(set(cands1)), len(set(cands2)), len(set(cands3)))
# # print(len(set(cands1) & set(cands2)))
# # print(len(set(cands1) & set(cands3)))
# # print(set(cands2) | set(cands1))


# print(cands1)
# # print(len(cands))
# # print(cands)


