from transformers import BertTokenizer
import tokenizations
import spacy
import benepar

class SpecialTokenError(Exception):
    pass

nlp = spacy.load("en_core_web_sm")
if spacy.__version__.startswith('2'):
    nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
else:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_clause_from_bert_token_idx(s: str, token_idx: int, truncate_first: bool, max_token_length=512):
    bert_tokenization = tokenizer.encode(s, add_special_tokens=True)
    if len(bert_tokenization) > max_token_length:
        if truncate_first:
            bert_tokenization = bert_tokenization[-max_token_length:]
        else:
            bert_tokenization = bert_tokenization[:max_token_length]
    bert_tokenization = tokenizer.convert_ids_to_tokens(bert_tokenization)
    doc = nlp(s)
    spacy_tokenization = [t.text for t in doc]
    bert2spacy, _ = tokenizations.get_alignments(bert_tokenization, spacy_tokenization)

    try:
        token = doc[bert2spacy[token_idx][0]]
    except IndexError:
        raise SpecialTokenError

    span = token
    #print(token.text)
    #print(list(doc.sents)[0]._.parse_string)
    if span._.parent is not None:
        span = token._.parent

    while span._.parent is not None and span._.labels[0] != 'S' and span._.labels[0] != 'SBAR':
        span = span._.parent

    SBAR_idx = 0
    broken = False
    constituents_ls = list(span._.constituents)
    for SBAR_idx in range(1, len(constituents_ls)):
        if 'SBAR' in constituents_ls[SBAR_idx]._.labels and constituents_ls[SBAR_idx].text != constituents_ls[0].text:
            broken = True
            break
    if broken:
        return constituents_ls[0].text.replace(constituents_ls[SBAR_idx].text, '')
    return constituents_ls[0].text

if __name__ == "__main__":
    s = 'In these cases of an emergency, passengers of the self-driving car may be overwhelmed or uneducated of how to respond in order to safely redirect the car.'
    token_idx = 30
    print(get_clause_from_bert_token_idx(s, token_idx, False))