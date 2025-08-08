import heapq
import json
import copy
from concurrent import futures
from collections import Counter, defaultdict
from typing import Counter, Optional, DefaultDict, Iterator
from .pre_tokenizer import PreTokenizer
from .pair_freq_heap import PairFreqHeap
from .utils import bytes_to_unicode
from .text_chunker import TextChunker
        

class BPETokenizer:
    
    MAX_WORKERS_THREAD_POOL = 32

    # --- Initialization Module ---

    def __init__(
        self,
        vocab: Optional[dict[int, bytes]] = None,
        merges: Optional[list[tuple[bytes, bytes]]] = None,
        special_tokens: Optional[list[str]] = None,
    ):
        self.vocab = vocab
        self.pairs_merged = merges
        self.special_tokens = special_tokens

        if self.special_tokens:
            self.vocab_size = len(vocab.keys()) + len(self.special_tokens)
            self._update_vocab(special_tokens)

        if self.vocab is not None:
            self.vocab_rev = {token: idx for idx, token in self.vocab.items()}
        
        if self.pairs_merged is not None:
            self.pairs_merged_prior = dict(zip(self.pairs_merged, range(len(self.pairs_merged))))
        
        # Create text chunker
        self.text_chunker = TextChunker()
        # Create pre-tokenizer
        self.pre_tokenizer = PreTokenizer(special_tokens)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[list[str]]=None):
        vocab = None
        pairs_merged = None

        # Read vocabulary from file
        with open(vocab_filepath, 'r') as f:
            vocab = json.load(f)
        vocab = {idx: token.encode('utf-8') for token, idx in vocab.items()}

        # Read merges from file
        with open(merges_filepath, 'r') as f:
            lines = f.readlines()
        pairs_merged = [tuple(line.split()) for line in lines]

        # Create tokenizer instance
        tokenizer = cls(vocab, pairs_merged, special_tokens)

        return tokenizer

    def _update_vocab(self, tokens: list[str | bytes]):
        idx = len(self.vocab)
        for token in tokens:
            token_bytes = token.encode('utf-8') if isinstance(token, str) else token
            if token_bytes in self.vocab.values():
                continue
            match token:
                case str():
                    self.vocab[idx] = token.encode('utf-8')
                    idx += 1
                case bytes():
                    self.vocab[idx] = token
                    idx += 1
                case _:
                    raise TypeError(f"Token must be str or bytes, but got {type(token)}")

    def _update_vocab_initial(self):
        idx = len(self.vocab)
        for i in range(256):
            if idx + i >= self.vocab_size:
                break
            self.vocab[i + idx] = bytes([i])


    # --- Training Module ---
    
    def _setup_train(self, input_path: str, vocab_size: int, max_num_chunks = 8):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.max_num_chunks = max_num_chunks

        # Vocabulary mapping index to its byte.
        self.vocab: dict[int, bytes] = dict()
        self._update_vocab(self.special_tokens)
        self._update_vocab_initial()

        self.word_freq = Counter()
        self.word_to_split: dict[str, list[bytes]] = dict()
        self.pair_to_words = defaultdict(Counter)
        self.pairs_freq_heap = PairFreqHeap()
        self.pairs_merged = list()        

    def _pre_tokenize_train(self) -> Counter:
        return self.pre_tokenizer.pre_tokenize_train_concurrency(self.input_path, self.pre_tokenize_parallel)
    
    def _pre_process(self):
        self.word_freq = self._pre_tokenize_train()

        for word, freq in self.word_freq.items():
            word_split = [bytes([letter]) for letter in word.encode('utf-8')]
            word_pairs = list(zip(word_split[:-1], word_split[1:]))
            
            # Count the pair frequency.
            # NOTE: there may be repeated pairs in one word
            word_pairs_counter = Counter()
            for pair in word_pairs:
                word_pairs_counter[pair] += freq

            # Update split word vocab and pair frequency
            self.word_to_split[word] = word_split
            self.pairs_freq_heap.counter_update(word_pairs_counter)

            # Updata pair to word mapping dict
            for pair in word_pairs:
                self.pair_to_words[pair][word] += 1
        
        # Build max heap for pairs
        self.pairs_freq_heap.counter_heapify()
    
    def _get_max_freq_pair(self) -> Optional[tuple[int, tuple[bytes, bytes]]]:
        return self.pairs_freq_heap.heap_pop()

    def _update_vocab_pairs_freq(self, pair):
        new_token = pair[0] + pair[1]
        updated_pairs = set()  # pairs that are deleted or newly formed

        for word in self.pair_to_words[pair]:
            if self.pair_to_words[pair][word] == 0:
                continue
            word_freq = self.word_freq[word]
            word_split = self.word_to_split[word]

            i = 0
            while i < len(word_split) - 1:
                if word_split[i] == pair[0] and word_split[i + 1] == pair[1]:
                    # Pair matched
                    word_split[i] = new_token
                    word_split.pop(i + 1)
                
                    # [a, b, c] => [a, bc], (a, b) => (a, bc)
                    if i > 0:
                        old_pair = (word_split[i - 1], pair[0])
                        new_pair = (word_split[i - 1], new_token)

                        # Updata pair frequency counter.
                        # NOTE: Heap is lazily updated.
                        self.pairs_freq_heap.increment_pair_count(old_pair, -word_freq)
                        self.pairs_freq_heap.increment_pair_count(new_pair, word_freq)
                        
                        # Record updated pairs for heap lazy updation.
                        updated_pairs.add(old_pair)
                        updated_pairs.add(new_pair)
                        
                        # Update pair to word map
                        self.pair_to_words[new_pair][word] += 1
                        self.pair_to_words[old_pair][word] -= 1

                    # [a, b, c] => [ab, c], (b, c) => (ab, c)
                    if i < len(word_split) - 1:
                        old_pair = (pair[1], word_split[i + 1])
                        new_pair = (new_token, word_split[i + 1])

                        self.pairs_freq_heap.increment_pair_count(old_pair, -word_freq)
                        self.pairs_freq_heap.increment_pair_count(new_pair, word_freq)
                        
                        updated_pairs.add(old_pair)
                        updated_pairs.add(new_pair)
                        
                        self.pair_to_words[new_pair][word] += 1
                        self.pair_to_words[old_pair][word] -= 1
                i += 1

            # Update word to split map
            self.word_to_split[word] = word_split
        
        # Delete the word map of the pair
        del self.pair_to_words[pair]
        self.pairs_freq_heap.remove_pair(pair)

        # Update max heap
        self.pairs_freq_heap.heap_update(updated_pairs)

    
    def train(self, input_path: str, vocab_size: int, pre_tokenize_prallel = 8):
        """
        Train the bpe tokenizer. Input file path and special tokens have been set in initialization.
        """

        # 1. Setup attributes for training.
        self._setup_train(input_path, vocab_size, pre_tokenize_prallel)

        # 2. Pre process: build up pairs frequency heap.
        self._pre_process()

        # 3. Construct token from pair with max frequency until vocabulary is full.
        vocab_idx = len(self.vocab)
        while vocab_idx < self.vocab_size:
            # Get pair with max frequency
            pair_wrapper = self._get_max_freq_pair()
            if pair_wrapper is None:
                break
            _, pair = pair_wrapper

            # Concatenate pairs to a new token and add it to vocabulary
            new_token = pair[0] + pair[1]
            self.vocab[vocab_idx] = new_token
            self.pairs_merged.append(pair)

            # Update pairs freqency
            self._update_vocab_pairs_freq(pair)

            vocab_idx += 1
        
        # 4. Update reverse vocabulary and merged pairs priority dict
        self.vocab_rev = {token: idx for idx, token in self.vocab}
        self.pairs_merged_prior = dict(zip(self.pairs_merged, range(len(self.pairs_merged))))
        
        return self.vocab, self.pairs_merged
    

    # --- Encoding Module ---

    def _pre_tokenize(self, text: str) -> list[str]:
        return self.pre_tokenizer.pre_tokenize_text(text)  
    
    def _word_to_tokens(self, word):
        # Exception for special tokens
        if self.special_tokens and word in self.special_tokens:
            return [self.vocab_rev[word.encode('utf-8')]]
        
        word_split = [bytes([letter]) for letter in word.encode('utf-8')]
        while True:
            if len(word_split) < 2:
                break
            word_pairs = zip(word_split[:-1], word_split[1:])
            pair_idx, pair = min(enumerate(word_pairs), key=lambda x: self.pairs_merged_prior.get(x[1], 1e10))
            if pair not in self.pairs_merged:
                break
            word_split[pair_idx] = word_split[pair_idx] + word_split[pair_idx + 1]
            word_split.pop(pair_idx + 1)
        
        return [self.vocab_rev[token] for token in word_split]
    
    def _encode_chunk(self, text: str, start: int, end: int):
        # 1. Pre tokenize text.
        words = self.pre_tokenizer.pre_tokenize_text(text, start, end)
        
        # 2. Transform pre-tokenized word to tokens in vocabulary according to merged pair.
        chunk_tokens = []
        for word in words:
            tokens = self._word_to_tokens(word)
            chunk_tokens.extend(tokens)
        
        return chunk_tokens

    def encode(self, text: str) -> list[int]:
        return self._encode_chunk(text, 0, len(text))
    
    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        for chunk in iterable:
            chunk_tokens =  self._encode_chunk(chunk, 0, len(chunk))
            for token in chunk_tokens:
                yield token

    def encode_concurrency(self, text: str, num_chunks = 16):
        """
        Encode text given by vocabulary and merged pairs.
        """
        
        # 1. Guess the boundaries for chunks
        boundaries = self.text_chunker.find_text_chunk_token_boundaries(text, num_chunks)
        
        # 2. Encode chunks in concurrency
        num_chunks = len(boundaries) - 1
        tokens = [None] * num_chunks
        future_tasks_index = dict()
        with futures.ThreadPoolExecutor(max_workers=self.MAX_WORKERS_THREAD_POOL) as executor:
            for i, (start, end) in enumerate(zip(boundaries[1:], boundaries[:-1])):
                future_task = executor.submit(self._encode_chunk, text, boundaries, start, end)
                future_tasks_index[future_task] = i
            
            for future_taks in futures.as_completed(future_tasks_index.keys()):
                result = future_taks.result()
                tokens[future_tasks_index[future_task]] = result
        
        # 3. Merge tokens from chunks
        merged_tokens = []
        for i in range(num_chunks):
            merged_tokens.append(tokens[i])
        
        return merged_tokens
    

    # --- Decoding Module ---
    
    def decode(self, ids: list[int]) -> str:
        text = b''
        for token_idx in ids:
                text += self.vocab[token_idx]
        text_str = text.decode('utf-8', errors='replace')
        return text_str


    # --- Output Module ---
    
    def dump_vocab(self, output_path: str):
        """
        Dump the vocabulary in .json format.
        """
        deocded_vocab = copy.deepcopy(self.vocab)
        for idx, token in deocded_vocab.items():
            deocded_vocab[idx] = token.decode('utf-8')
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(deocded_vocab, f, ensure_ascii=False, indent=4)

    def dump_pairs_merged(self, output_path: str):
        """
        Dump the vocabulary in .txt format.
        """
        with open(output_path, "w", encoding='utf-8') as f:
            for token1, token2 in self.pairs_merged:
                f.write(token1.decode('utf-8', errors='ignore') + " " + token2.decode('utf-8', errors='ignore') + "\n")