import numpy as np
import string
from collections import Counter


def read_iob_file(file_path, ignore_punctuation=False):
    all_tokens = []
    sentence_tokens = []

    all_pos_tags = []
    sentence_pos_tags = []
    
    all_chunk_tags = []
    sentence_chunk_tags = []
    
    all_entity_tags = []
    sentence_entity_tags = []
    

    with open(file_path) as file:
        for line in file:
            line = line.rstrip()
            
            if line.startswith("-DOCSTART-") or not line:
                if sentence_tokens:
                    all_tokens.append(sentence_tokens)
                    sentence_tokens = []
                    all_pos_tags.append(sentence_pos_tags)
                    sentence_pos_tag = []
                    all_chunk_tags.append(sentence_chunk_tags)
                    sentence_chunk_tag = []
                    all_entity_tags.append(sentence_entity_tags)
                    sentence_entity_tags = []
                continue

            token, pos_tag, chunk_tag, entity_tag = line.split(" ")
            
            if ignore_punctuation and token in string.punctuation:
                continue
                
            sentence_tokens.append(token)
            sentence_pos_tags.append(pos_tag)
            sentence_chunk_tags.append(chunk_tag)
            sentence_entity_tags.append(entity_tag)
    
    return {
        "tokens": all_tokens,
        "pos_tags": all_pos_tags,
        "chunk_tags": all_chunk_tags,
        "entity_tags": all_entity_tags
    }

def preprocess_tokens(tokens, vocab_size, sequence_length):
    # Tokens frequency
    vocab_counter = Counter()
    for sentence in tokens:
        vocab_counter.update(sentence)
    
    # Take the most common vocab_size - 2 tokens, because
    # of padding and unknown tokens
    most_common_tokens = vocab_counter.most_common(vocab_size - 2)
    vocab = [token for token, _ in most_common_tokens]

    tokenToIdx = {
        "[PAD]": 0,
        "[UNK]": 1
    }
    tokenToIdx.update({
        token: idx + 2 for idx, token in enumerate(vocab)
    })
    
    v_tokens = vectorize_tokens(tokens, tokenToIdx, sequence_length)
    return v_tokens, tokenToIdx


def vectorize_tokens(tokens, tokenToIdx, sequence_length):
    v_tokens = []
    for sentence in tokens:
        padded_sentence = []
        for i in range(sequence_length):
            if i < len(sentence):
                token = sentence[i].lower()
                idx = tokenToIdx.get(token, tokenToIdx["[UNK]"])
                padded_sentence.append(idx)
            else:
                padded_sentence.append(tokenToIdx["[PAD]"])
        v_tokens.append(padded_sentence)
    return np.array(v_tokens, dtype=np.int64)


def preprocess_entity_tags(entity_tags, sequence_length):
    tags = set()

    for sentence_entity_tags in entity_tags:
        tags.update(sentence_entity_tags)
    tags = sorted(tags)

    entityToIdx = {
        "[PAD]": 0
    }
    entityToIdx.update({
        entity: idx + 1 for idx, entity in enumerate(tags)
    })

    v_entity_tags = vectorize_entity_tags(entity_tags, entityToIdx, sequence_length)
    return v_entity_tags, tags


def vectorize_entity_tags(entity_tags, entityToIdx, sequence_length):
    v_entity_tags = []
    for sentence_entity_tags in entity_tags:
        padded_sentence_entity_tags = []
        for i in range(sequence_length):
            if i < len(sentence_entity_tags):
                entity_tag = sentence_entity_tags[i]
                idx = entityToIdx[entity_tag]
                padded_sentence_entity_tags.append(idx)
            else:
                padded_sentence_entity_tags.append(entityToIdx["[PAD]"])
        v_entity_tags.append(padded_sentence_entity_tags)
    return np.array(v_entity_tags, dtype=np.int64)
