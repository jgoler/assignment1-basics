import regex as re
from collections import Counter

INITIAL_BYTE_VOCAB_LEN = 256
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(input_path:str, vocab_size: int, special_tokens: list[str]):
    text = ""
    with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
    vocab = build_starting_vocab(special_tokens, vocab_size)
    merge_list = []
    counts = convert_string_dict_to_byte_dict(pre_tokenize(text, special_tokens))
    vocab_cur_size = len(vocab)
    
    while (vocab_cur_size < vocab_size):
        left, right = find_most_occurring_byte_pair(counts)
        counts = merge_most_occurring_byte_pair(left, right, counts)
        vocab[vocab_cur_size] = left + right
        merge_list.append((left, right))
        vocab_cur_size += 1

    return (vocab, merge_list)


def construct

'''
ARGS:
input_path: string -> string to the txt file
vocab_size: maximum vocabulary size of tokens that we build up to
special_tokens: list[str] of strings we split on

RETURN:
return 
'''

def split_special_tokens(text: str, special_tokens: list[str]):
    if not special_tokens:
        return [text]
    # first escape anything that may cause an issue later
    escaped_special_tokens = []
    for i in range(len(special_tokens)):
        escaped_special_tokens.append(re.escape(special_tokens[i]))

    
    pattern_list_text = "|".join(escaped_special_tokens)
    return re.split(pattern_list_text, text)


def pre_tokenize(text: str, special_tokens: list[str]):

    chunks = split_special_tokens(text, special_tokens)

    counts = Counter()
    for chunk in chunks:
        matches = re.finditer(PAT, chunk)
        counts.update(match.group() for match in matches)

    return counts

def find_most_occurring_byte_pair(pretokenized_dict: dict[bytes]):
    assert pretokenized_dict, "Please make sure to pass a nonempty dict"
    byte_pair_counts = Counter()
    for pre_token_byte_tuple in pretokenized_dict.keys():
        for left, right in zip(pre_token_byte_tuple, pre_token_byte_tuple[1:]):
            byte_pair_counts[(left, right)] += pretokenized_dict[pre_token_byte_tuple]
    new_vocab_token = max(byte_pair_counts, key=lambda k: (byte_pair_counts[k], k))
    return new_vocab_token


def merge_most_occurring_byte_pair(left_byte: bytes, right_byte: bytes, pre_token_bytes_dict: dict):
    new_dict = Counter()
    for pre_token_byte_tuple in pre_token_bytes_dict.keys():
        new_tuple = create_new_tuple_given_bytes(left_byte, right_byte, pre_token_byte_tuple)
        new_dict[new_tuple] += pre_token_bytes_dict[pre_token_byte_tuple]
    return new_dict




def create_new_tuple_given_bytes(left: bytes, right: bytes, t: tuple):
    new_tuple_as_list = []
    i = 0
    j = 1
    while j < len(t):
        if t[i] == left and t[j] == right:
            new_tuple_as_list.append(t[i] + t[j])
            i += 2
            j += 2
        else:
            new_tuple_as_list.append(t[i])
            i += 1
            j += 1
    if i < len(t):
        new_tuple_as_list.append(t[i])

    return tuple(new_tuple_as_list)





'''
ARGS
counts_dict: dict -> dictionary of strings 

RETURNS
dictionary of tuples of bytes
'''
def convert_string_dict_to_byte_dict(counts_dict: dict):
    new_dict = {}
    for key, value in counts_dict.items():
        key_as_bytes = key.encode('utf-8')
        new_key = tuple(bytes([byte]) for byte in key_as_bytes)
        new_dict[new_key] = value
    return new_dict

def build_starting_vocab(special_tokens, vocab_size):
    # check that the vocab_size is at least as large as the initial byte vocabulary and special tokens
    assert vocab_size >= len(special_tokens) + INITIAL_BYTE_VOCAB_LEN, "vocab_size is too small."
    d = {}
    for i in range(INITIAL_BYTE_VOCAB_LEN):
        d[i] = bytes([i])

    num_special_tokens = len(special_tokens)
    for j in range(INITIAL_BYTE_VOCAB_LEN, INITIAL_BYTE_VOCAB_LEN + num_special_tokens):
        d[j] = special_tokens[j - INITIAL_BYTE_VOCAB_LEN].encode('utf-8')

    return d

if __name__ == "__main__":
    train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 300, [])


