import os
import itertools
import regex as re
from tqdm import tqdm
from typing import BinaryIO, DefaultDict, Counter, Any, Optional
from concurrent import futures
from collections import defaultdict, Counter

from .text_chunker import TextChunker


class PreTokenizer:
    NUM_WORKERS = 16

    def __init__(self, special_tokens: Optional[list[str]] = None):
        self.special_tokens = special_tokens

        if self.special_tokens:
            self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.delimiter_pattern = '|'.join(re.escape(token) for token in self.special_tokens)

        self.pta_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        self.text_chunker = TextChunker()

    def pre_tokenize_train(
        self,
        file_path: str,
        start: int,
        end: int,
    ) -> Counter:
        # 1. Read file chunk from start to end
        with open(file_path, "rb") as file: 
            file.seek(start)
            chunk = file.read(end - start).decode("utf-8", errors="ignore")

        # 2. Split chunk to chunks not having special tokens
        if self.special_tokens:
            split_chunks = re.split(self.delimiter_pattern, chunk)
        else:
            split_chunks = [chunk]

        # 3. Count the words in split chunks
        freq = Counter()
        for chunk in split_chunks:
            tokens = [match.group(0) for match in re.finditer(self.pta_pattern, chunk)]
            freq.update(tokens)
        
        return freq   
    
    
    @staticmethod
    def _merge_freq(freq: list[Counter]) -> Counter:
        """
        Merge all the freq.
        """
        assert len(freq) > 0
        merged_freq = freq[0]

        for i in range(1, len(freq)):
            merged_freq += freq[i]
            freq[i].clear()
        
        return merged_freq

    
    def pre_tokenize_train_concurrency(self, input_path: str, num_chunks = 64) -> Counter:
        freq = []

        with open(input_path, "rb") as file:
            boundaries = self.text_chunker.find_file_chunk_special_token_boundaries(file, num_chunks, b"<|endoftext|>")

        tasks_id = list(range(len(boundaries) - 1))
        with tqdm(total=len(tasks_id), desc='running') as pbar:
            with futures.ThreadPoolExecutor(max_workers=self.NUM_WORKERS) as executor:
                future_tasks = []

                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    # Run pre-tokenization on your chunk and store the counts for each pre-token
                    future_tasks.append(executor.submit(self._pre_tokenize_train, input_path, start, end))
                
                for future_task in future_tasks:
                    result = future_task.result()
                    freq.append(result)
                    pbar.update(1)
        
        merged_freq = self._merge_freq(freq)

        return merged_freq
    
    def pre_tokenize_file(
        self,
        file_path: str,
        start: int,
        end: int,        
    ):
        # Read file chunk from start to end
        with open(file_path, "rb") as file: 
            file.seek(start)
            chunk = file.read(end - start).decode("utf-8", errors="ignore")
            
        # Split chunk to chunks not having special tokens
        if self.special_tokens:
            split_chunks = re.split(r'({self.delimiter_pattern})', chunk)
        else:
            split_chunks = [chunk]

        # Count the words in split chunks
        tokens = []
        for chunk in split_chunks:
            tokens.extend([match.group(0) for match in re.finditer(self.pta_pattern, chunk)])
        
        return tokens

    def pre_tokenize_text(
        self,
        text: str,
        start: int,
        end: int,        
    ):
        chunk = text[start: end]
            
        # Split chunk to chunks not having special tokens
        if self.special_tokens:
            split_chunks = re.split(f'({self.delimiter_pattern})', chunk)
        else:
            split_chunks = [chunk]

        # Count the words in split chunks
        tokens = []
        for chunk in split_chunks:
            if self.special_tokens and chunk in self.special_tokens:
                tokens.append(chunk)
            else:
                tokens.extend([match.group(0) for match in re.finditer(self.pta_pattern, chunk)])
        
        return tokens

    def pre_tokenize_file_concurrency(
        self, 
        input_path: str,
        token_dict: dict[bytes, int], 
        num_chunks = 64
    ) -> list[str]:
        tokens = []
        
        with open(input_path, "rb") as file:
            boundaries = self.text_chunker.find_file_chunk_token_boundaries(file, num_chunks, token_dict)

        num_chunks = len(boundaries) - 1
        tokens = [None] * num_chunks
        future_tasks_index = dict()
        with tqdm(total=num_chunks, desc='handling chunks') as pbar:
            with futures.ThreadPoolExecutor(max_workers=self.NUM_WORKERS) as executor:
                for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                    # Run pre-tokenization on your chunk and store the counts for each pre-token
                    future_task = executor.submit(self._pre_tokenize, input_path, start, end)
                    future_tasks_index[future_task] = i
                
                for future_task in futures.as_completed(future_tasks_index.keys()):
                    result = future_task.result()
                    tokens[future_tasks_index[future_task]] = result
                    pbar.update(1)
        
        if tokens.find(None) > -1:
            raise ValueError
        
        return tokens

    def pre_tokenize_text_concurrency(
        self, 
        text: str,
        token_dict: dict[bytes, int], 
        num_chunks = 64
    ) -> list[str]:
        tokens = []
        
        boundaries = self.text_chunker.find_text_chunk_token_boundaries(text, num_chunks, token_dict)

        num_chunks = len(boundaries) - 1
        tokens = [None] * num_chunks
        future_tasks_index = dict()
        with tqdm(total=num_chunks, desc='handling chunks') as pbar:
            with futures.ThreadPoolExecutor(max_workers=self.NUM_WORKERS) as executor:
                for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                    # Run pre-tokenization on your chunk and store the counts for each pre-token
                    future_task = executor.submit(self._pre_tokenize_text, text, start, end)
                    future_tasks_index[future_task] = i
                
                for future_task in futures.as_completed(future_tasks_index.keys()):
                    result = future_task.result()
                    tokens[future_tasks_index[future_task]] = result
                    pbar.update(1)
        
        if tokens.find(None) > -1:
            raise ValueError
        
        return tokens