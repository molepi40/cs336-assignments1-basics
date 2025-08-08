import os
import itertools
import regex as re
from tqdm import tqdm
from typing import BinaryIO, DefaultDict, Counter, Any, Optional
from concurrent import futures
from collections import defaultdict, Counter

class TextChunker:
    """
    Chunk the text to desired number of chunks given special token or
    avoiding chunk boundaries crossing token.
    """
    def __init__(self, min_chunk_size = 4096):
        pass
    
    def find_file_chunk_special_token_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    @staticmethod
    def _find_one_token_bytes(
        data: bytes, 
        sorted_token_dict: dict[bytes, int]
    ) -> Optional[bytes]:
        """
        Find the longest matching token from the beginning of the data.
        This helper function attempts to decode one token greedily.
        """
        # Check for tokens from longest to shortest for a greedy match
        # Note: A more optimized approach would be using a Trie (prefix tree).
        # For simplicity, we sort keys by length here.
        for token in sorted_token_dict.keys():
            if data.startswith(token):
                return token
        return None

    def find_file_chunk_token_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        token_dict: dict[bytes, int],
    ) -> list[int]:
        """
        Chunks the file into parts that can be tokenized independently.

        This function identifies chunk boundaries that do not split any tokens.
        It may return fewer chunks if boundaries end up overlapping.

        Args:
            file: The binary file stream to process.
            desired_num_chunks: The target number of chunks.
            token_dict: A dictionary mapping token bytes to their integer IDs.

        Returns:
            A sorted list of unique byte offsets that represent the chunk boundaries.
        """
        # 1. Get total file size and calculate max token length.
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        
        if file_size == 0:
            return [0]

        # The max_token_len determines how far back we need to search for a valid token start.
        if not token_dict:
            raise ValueError("token_dict cannot be empty.")
        sorted_tokens = sorted(token_dict.keys(), key=len, reverse=True)
        max_token_len = len(sorted_tokens[0])

        # 2. Make initial guesses for chunk boundary locations, uniformly spaced.
        chunk_size = file_size // desired_num_chunks
        # The boundaries list includes the start (0) and end (file_size) of the file.
        boundaries = [i * chunk_size for i in range(desired_num_chunks)]
        boundaries.append(file_size)

        # 3. Adjust each internal boundary to align with a token start.
        # We don't need to adjust the first (0) or the last (file_size) boundary.
        for i in range(1, len(boundaries) - 1):
            estimated_boundary = boundaries[i]
            
            # We search backwards from the estimated boundary to find the beginning
            # of the token that this boundary might have split.
            # We only need to search back up to max_token_len bytes.
            found_safe_boundary = False
            for offset in range(max_token_len + 1):
                start_pos = estimated_boundary - offset
                
                # Ensure we don't seek before the beginning of the file.
                if start_pos < 0:
                    start_pos = 0

                # Seek to the potential start of a token and read a chunk.
                file.seek(start_pos)
                # Read enough bytes to contain at least the longest possible token.
                chunk_to_inspect = file.read(max_token_len)
                
                if not chunk_to_inspect:
                    break # Reached end of file

                # Try to decode one token from this position.
                token = self._find_one_token_bytes(chunk_to_inspect, token_dict)

                # If a token is found, check if it crosses our estimated boundary.
                if token is not None:
                    # If the token we found at `start_pos` extends to or beyond
                    # our original `estimated_boundary`, then `start_pos` is the
                    # true beginning of the token we split. This is a safe boundary.
                    if start_pos + len(token) >= estimated_boundary:
                        boundaries[i] = start_pos
                        found_safe_boundary = True
                        break # Move to the next boundary

        # 4. Return sorted unique boundaries.
        # The set() handles cases where adjusted boundaries become identical.
        final_boundaries = sorted(list(set(boundaries)))

        return final_boundaries

    def find_text_chunk_token_boundaries(
        self,
        text: str | bytes,
        desired_num_chunks: int,
        token_dict: dict[bytes, int],
    ) -> list[int]:
        """
        Chunks a large string into parts that can be tokenized independently.

        This function identifies chunk boundaries (indices) that do not split any tokens.
        It's designed for when the entire text content is already in memory as a string.

        Args:
            text: The string to process.
            desired_num_chunks: The target number of chunks.
            token_dict: A dictionary mapping token strings to their integer IDs.

        Returns:
            A sorted list of unique character indices that represent the chunk boundaries.
        """
        # 1. Get total text length and calculate max token length.
        if isinstance(text, str):
            text = text.encode('utf-8')
        elif not isinstance(text, bytes):
            raise TypeError(f'Text must be str or bytes, not {type(text)}')
        
        text_len = len(text)
        
        if text_len == 0:
            return [0]

        if not token_dict:
            raise ValueError("token_dict cannot be empty.")
        # The length here is in characters, not bytes.
        sorted_token_dict = sorted(token_dict, key=len, reverse=True)
        max_token_len = max(len(token) for token in token_dict.keys())

        # 2. Make initial guesses for chunk boundary locations, uniformly spaced.
        chunk_size = min(text_len // desired_num_chunks, self.MIN_CHUNK_SIZE)
        boundaries = [i * chunk_size for i in range(desired_num_chunks)]
        boundaries.append(text_len)

        # 3. Adjust each internal boundary to align with a token start.
        for i in range(1, len(boundaries) - 1):
            estimated_boundary = boundaries[i]
            
            for offset in range(max_token_len + 1):
                start_pos = estimated_boundary - offset
                
                if start_pos < 0:
                    start_pos = 0

                # Use string slicing instead of file I/O. This is much faster.
                chunk_to_inspect = text[start_pos : start_pos + max_token_len]
                
                if not chunk_to_inspect:
                    break 

                token = self._find_one_token_bytes(chunk_to_inspect, sorted_token_dict)

                if token is not None:
                    # The core validation logic is identical to the bytes version.
                    if start_pos + len(token) >= estimated_boundary:
                        boundaries[i] = start_pos
                        break 

        # 4. Return sorted unique boundaries.
        final_boundaries = sorted(list(set(boundaries)))

        return final_boundaries