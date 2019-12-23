import os
import random
import logging
import json

from flask import Flask, request

import torch
from transformers.tokenization_bert import BertTokenizer
from pyknp import Juman

from model import BertMouth
from data import make_dataloader

MAX_SEQ_LEN = 40
MAX_ITER = 5

bert_model_path = './rap_bert_model'

app = Flask(__name__)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

tokenizer = BertTokenizer.from_pretrained(bert_model_path, do_lower_case=False,
                                          tokenize_chinese_chars=False)
juman = Juman()

device = 'cpu'

model_state_dict = torch.load(os.path.join(bert_model_path, "pytorch_model.bin"),
                              map_location=device)
bert_model = BertMouth.from_pretrained(bert_model_path,
                                  state_dict=model_state_dict,
                                  num_labels=tokenizer.vocab_size)
bert_model.to(device)


def initialization_text(tokenizer, length, fix_word):
    except_tokens = ["[MASK]", "[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    except_ids = [tokenizer.vocab[token] for token in except_tokens]
    candidate_ids = [i for i in range(tokenizer.vocab_size)
                     if i not in except_ids]

    init_tokens = []
    init_tokens.append(tokenizer.vocab["[CLS]"])
    for _ in range(length):
        init_tokens.append(random.choice(candidate_ids))
    init_tokens.append(tokenizer.vocab["[SEP]"])

    return init_tokens


def generate(tokenizer, device, max_iter=10, length=50, max_length=128,
             model=None, fix_word=None):
    generated_token_ids = initialization_text(tokenizer, length, fix_word)

    if fix_word:
        tokenized_fix_word = tokenizer.tokenize(fix_word)
        fix_word_pos = random.randint(1,
                                      length - len(tokenized_fix_word))
        fix_word_interval = set(range(fix_word_pos,
                                      fix_word_pos + len(tokenized_fix_word)))
        for i in range(len(tokenized_fix_word)):
            generated_token_ids[fix_word_pos + i] = \
                tokenizer.convert_tokens_to_ids(tokenized_fix_word[i])

    else:
        fix_word_interval = []

    input_type_id = [0] * max_length
    input_mask = [1] * len(generated_token_ids)
    while len(input_mask) < max_length:
        generated_token_ids.append(0)
        input_mask.append(0)

    generated_token_ids = torch.tensor([generated_token_ids],
                                       dtype=torch.long).to(device)
    input_type_id = torch.tensor(
        [input_type_id], dtype=torch.long).to(device)
    input_mask = torch.tensor([input_mask], dtype=torch.long).to(device)

    pre_tokens = generated_token_ids.clone()
    for _ in range(max_iter):
        for j in range(length):
            if fix_word_interval:
                if j + 1 in fix_word_interval:
                    continue

            generated_token_ids[0, j + 1] = tokenizer.vocab["[MASK]"]
            logits = bert_model(generated_token_ids,
                           input_type_id, input_mask)[0]
            sampled_token_id = torch.argmax(logits[j + 1])
            generated_token_ids[0, j + 1] = sampled_token_id
        sampled_sequence = [tokenizer.ids_to_tokens[token_id]
                            for token_id in generated_token_ids[0].cpu().numpy()]
        sampled_sequence = "".join([token[2:] if token.startswith("##") else token
                                    for token in sampled_sequence[1:length + 1]])
        if torch.equal(pre_tokens, generated_token_ids):
            break
        pre_tokens = generated_token_ids.clone()
    logger.info("sampled sequence: {}".format(sampled_sequence))
    return sampled_sequence


def get_random_norm(text):
    j = juman.analysis(text)
    norms = set()
    for mrph in j.mrph_list():
        if mrph.hinsi == '名詞':
            norms.add(mrph.midasi)
    if norms:
        return random.choice(list(norms))

    return None

@app.route('/')
def hello():
    return 'Hello world!'

@app.route('/rap', methods=['POST'])
def rap():
    if request.form:
        verse = request.form['verse']
    elif request.data:
        jsons = str(request.data, encoding='utf-8')
        data = json.loads(jsons)
        verse = data['verse']

    fix_word = get_random_norm(verse)
    logger.info(f"fix_word: {fix_word}")

    verse_len = len(verse)
    seq_length = verse_len
    if verse_len < 15:
        seq_length = verse_len * 2
    seq_length = min(MAX_SEQ_LEN, seq_length)

    # TODO: use verse for fix_word
    gen_txt = generate(tokenizer, device, max_iter=MAX_ITER,
                       length=seq_length, model=bert_model_path,
                       fix_word=fix_word)

    return gen_txt

if __name__ == '__main__':
    app.run(host='0.0.0.0')
