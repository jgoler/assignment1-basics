"""Your byte-level BPE tokenizer. You build this file up over the stages in learning/README.md.

The course tests (tests/test_train_bpe.py, tests/test_tokenizer.py) and the stage checker
(learning/check.py) import from this module, so keep the names below. Replace each
`raise NotImplementedError` as you reach that stage. The finished reference implementation is
in tokenizer_reference.py; open it only after you have passed a stage's checks.
"""

from collections.abc import Iterable, Iterator
from collections import Counter
import regex
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# Stage 1 ------------------------------------------------------------------------------
class ByteTokenizer:
    """Vocabulary of exactly the 256 byte values: token ID == byte value."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))
        

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


# Stage 2 ------------------------------------------------------------------------------
def count_pretokens(text: str, special_tokens: list[str]) -> dict[bytes, int]:
    """Map each pre-token (as UTF-8 bytes) to how many times it occurs in `text`.

    Special tokens split the text into independent segments and are not counted themselves.
    """
    segments = split_on_special_tokens(text, special_tokens)
    return count_pretokens_as_bytes(segments)

def pretokenize(segment: str) -> list[str]:
    """Step 4. The pre-tokens of a segment that contains no special tokens: every match of PAT, in order.
    pretokenize("some text that i'll pre-tokenize") -> see spec section 2.4 for the exact output.
    Use regex.finditer and match.group() (the spec asks for finditer; findall works too for now)."""
    matches = regex.finditer(PAT, segment)
    return [match.group() for match in matches]

def escape_token(token: str) -> str:
    """Step 1. A regex pattern that matches `token` literally, even though it contains `|`.
    In the REPL compare regex.split("<|a|>", "x<|a|>y") with regex.split(escape_token("<|a|>"), "x<|a|>y")."""
    return regex.escape(token)


def special_token_pattern(special_tokens: list[str]) -> str:
    """Step 2. One pattern matching ANY of the tokens: the escaped tokens joined with "|" (regex "or").
    Put longer tokens first (drill 15): regex tries alternatives left to right and takes the first that
    matches, so with ["<|a|>", "<|a|><|a|>"] in that order the double token would never be found."""
    special_tokens = [escape_token(token) for token in special_tokens]
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    return "|".join(special_tokens)

def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """Step 3. The pieces of `text` between special tokens, in order, specials removed.
    With no special tokens, the whole text is the single piece.
    split_on_special_tokens("a<|x|>b<|x|>", ["<|x|>"]) -> ["a", "b", ""]"""
    if not special_tokens:
        return [text]
    token_pattern = special_token_pattern(special_tokens)
    return regex.split(token_pattern, text)

def count_pretokens_as_bytes(segments: list[str]) -> Counter:
    """Step 5. Pre-tokenize every segment and count each pre-token, keyed by its UTF-8 bytes (drill 2).
    count_pretokens_as_bytes(["a a", "a"]) -> Counter({b"a": 2, b" a": 1})"""
    pretoken_dict = Counter()
    for segment in segments:
        segment_pre_tokens = pretokenize(segment)
        for pre_token in segment_pre_tokens:
            pretoken_dict[pre_token.encode("utf-8")] += 1
    return pretoken_dict


# Stage 3: naive BPE training, built from small pieces -----------------------------------
# Build and check these in order: uv run python learning/check.py 3a, 3b, ... 3f.

Pair = tuple[bytes, bytes]
Word = tuple[bytes, ...]  # a pre-token as a sequence of tokens, e.g. (b"t", b"h", b"e") or (b"th", b"e")


def pretokens_to_words(pretoken_counts: dict[bytes, int]) -> dict[Word, int]:
    """3a. Re-key the stage 2 counts so each pre-token becomes a tuple of one-byte `bytes` (drill 11)."""
    word_counter = Counter()
    for pretoken in pretoken_counts:
        word = tuple(bytes([value]) for value in pretoken)
        word_counter[word] += pretoken_counts[pretoken]
    return word_counter



def count_pairs(words: dict[Word, int]) -> dict[Pair, int]:
    """3b. Total count of every adjacent pair across all words, weighted by word count (drill 12)."""
    pairs_dict = Counter()
    for word in words:
        for (byte1, byte2) in zip(word, word[1:]):
            pairs_dict[(byte1, byte2)] += words[word]
    return pairs_dict


def best_pair(pair_counts: dict[Pair, int]) -> Pair:
    """3c. The most frequent pair; ties go to the lexicographically greater pair (drill 13)."""
    return max(pair_counts, key=lambda x: (pair_counts[x], x))


def merge_word(word: Word, pair: Pair) -> Word:
    """3d. Replace each non-overlapping occurrence of `pair` in `word` with the two tokens joined (drill 14)."""
    result = []
    i = 0
    while i < len(word) - 1:
        if tuple([word[i], word[i + 1]]) == pair:
            result.append(word[i] + word[i + 1])
            i += 2
        else:
            result.append(word[i])
            i += 1
    if (i < len(word)):
        result.append(word[i])
    return tuple(result)
    


def merge_words(words: dict[Word, int], pair: Pair) -> dict[Word, int]:
    """3e. Apply `merge_word` to every word, keeping each word's count."""
    new_words = {}
    for word in words:
        new_words[merge_word(word, pair)] = words[word]
    return new_words


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[Pair]]:
    """3f. Read the file, count pre-tokens (stage 2), convert them (3a), then repeat 3b -> 3c -> 3e
    until the vocabulary is full. Stage 5 makes this fast without changing its results."""
    vocab = {i: bytes([i]) for i in range(256)}
    pair_list = []
    for i in range(len(special_tokens)):
        vocab[i + 256] = special_tokens[i].encode("utf-8")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    pretokens_dict = count_pretokens(text, special_tokens)
    word_counts = pretokens_to_words(pretokens_dict)
    for i in range(vocab_size - len(special_tokens) - 256):
        pairs = count_pairs(word_counts)
        if not len(pairs):
            break
        winner = best_pair(pairs)
        pair_list.append(winner)
        word_counts = merge_words(word_counts, winner)
        vocab[i + 256 + len(special_tokens)] = winner[0] + winner[1]

    return tuple([vocab, pair_list])




# Stage 4: encoding and decoding ----------------------------------------------------------
# Check in order: uv run python learning/check.py 4a, 4b, ... 4e.   Stage 6 adds encode_iterable.
class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Pair],
        special_tokens: list[str] | None = None,
    ) -> None:
        """4a. Store the inputs. Build `self.token_to_id`, the inverse of vocab (drill 20). Append any
        special token missing from the vocab with a fresh ID. Build `self.merge_ranks`: pair -> its
        position in `merges`, so "earliest merge" becomes "smallest rank"."""
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.vocab.values():
                self.vocab[len(self.vocab)] = token_bytes

        self.token_to_id = {token: idx for idx, token in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}



    def split_on_special_tokens(self, text: str) -> list[str]:
        """4b. Cut `text` into a list where every special token is its own element and the text between
        them keeps its order. When special tokens overlap, the longest wins (drill 15). Drop empty strings.
        "a<|endoftext|>b" -> ["a", "<|endoftext|>", "b"]"""
        if not self.special_tokens:
            return [text]
        ordered_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        special_token_pattern = "|".join(["(" + regex.escape(special_token) + ")" for special_token in ordered_special_tokens])
        split_text = regex.split(special_token_pattern, text)
        return [t for t in split_text if t]


    def encode_pretoken(self, pretoken: bytes) -> list[int]:
        """4c. Apply the merges to ONE pre-token: start from its one-byte tokens (3a), repeatedly merge the
        pair with the smallest rank that is present (drill 16, then 3d), stop when none applies, then map
        tokens to IDs."""
        word = tuple(bytes([b]) for b in pretoken)
        while True:
            pair, smallest_rank = None, float("inf")
            for left, right in zip(word, word[1:]):
                rank = self.merge_ranks.get((left, right))
                if rank is not None and rank < smallest_rank:
                    pair, smallest_rank = (left, right), rank
            if pair is None:
                break
            word = merge_word(word, pair)
        return [self.token_to_id[t] for t in word]

    def encode(self, text: str) -> list[int]:
        """4d. 4b, then for each segment: a special token becomes one ID, other text is pre-tokenized
        (stage 2's regex) and each pre-token goes through 4c."""
        tokenized = []
        split_text = self.split_on_special_tokens(text)
        for i in range(len(split_text)):
            if split_text[i] in self.special_tokens:
                tokenized.append(self.token_to_id[split_text[i].encode("utf-8")])
            else:
                pretokens = pretokenize(split_text[i])
                for pretoken in pretokens:
                    tokenized.extend(self.encode_pretoken(pretoken.encode("utf-8")))
        return tokenized
                

    def decode(self, ids: list[int]) -> str:
        """4d. Look up each ID's bytes, join them (drill 10), decode (stage 1)."""
        return b"".join([self.vocab[id] for id in ids]).decode("utf-8", errors="replace")


    def save(self, vocab_filepath: str, merges_filepath: str) -> None:
        """4e. Write vocab and merges to disk in any format you like, as long as `from_files` reads it back.
        Note: bytes are not JSON; decide how you'll represent them."""
        vocab_json = {idx: list(token) for idx, token in self.vocab.items()}
        merges_json = [[list(left), list(right)] for left, right in self.merges]

        with open(vocab_filepath, "w", encoding="utf-8") as f:
            json.dump(vocab_json, f)
        with open(merges_filepath, "w", encoding="utf-8") as f:
            json.dump(merges_json, f)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """4e. Inverse of `save`."""
        loaded_vocab = {}
        loaded_merges = []
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        for key in vocab:
            loaded_vocab[int(key)] = bytes(vocab[key])
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges = json.load(f)
        for left, right in merges:
            left = bytes(left)
            right = bytes(right)
            loaded_merges.append((left, right))
        return cls(loaded_vocab, loaded_merges, special_tokens)


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """6a-6c. A generator (drill 18) yielding the IDs of the concatenated chunks, never holding
        more than a chunk or so in memory, and yielding exactly what `encode` would on the whole text."""
        raise NotImplementedError
