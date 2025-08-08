import heapq
from collections import Counter, defaultdict
from typing import Counter, Optional, DefaultDict

class PairFreqWrapper:
    def __init__(self, data: tuple):
        if not isinstance(data, tuple):
            raise TypeError
        self.data = data

    def __lt__(self, other):
        if self.data[0] == other.data[0]:
            return self.data[1] > other.data[1]
        else:
            return self.data[0] > other.data[0]

    def __iter__(self):
        return iter(self.data)
    
    def __getitem__(self, index: int):
        assert index >= 0 and index < 2
        return self.data[index]
    
class PairFreqHeap:
    def __init__(self, counter: DefaultDict[tuple, int] = None):
        self.counter = counter if counter is not None else Counter()
        self.heap = list()
        
        # Create heap for counter
        if self.counter:
            self.heap = [PairFreqWrapper((freq, pair)) for pair, freq in self.counter.items()]
            heapq.heapify(self.heap)
    
    def counter_heapify(self):
        self.heap = [PairFreqWrapper((freq, pair)) for pair, freq in self.counter.items()]
        heapq.heapify(self.heap)        

    def _clean_heap(self):
        """
        Clean the invalid elements in heap.
        """
        # keys not in counter
        while self.heap and self.heap[0][1] not in self.counter:
            heapq.heappop(self.heap)
        
        # value not equal that of counter
        while self.heap and self.counter[self.heap[0][1]] != self.heap[0][0]:
            heapq.heappop(self.heap)
    
    def heap_pop(self) -> Optional[tuple[int, tuple[bytes, bytes]]]:
        self._clean_heap()
        if self.heap:
            freq, pair = heapq.heappop(self.heap)
            assert freq == self.counter[pair]
            del self.counter[pair]
            return freq, pair
        else:
            return None
    
    def remove_pair(self, pair):
        if pair in self.counter:
            del self.counter[pair]
            self._clean_heap()
    
    def increment_pair_count(self, pair, inc_cnt):
        """
        Increment the value for key, but not update heap instantly.
        """
        self.counter[pair] += inc_cnt
    
    def counter_update(self, other_counter: Counter):
        self.counter += other_counter
    
    def heap_update(self, pairs):
        self._clean_heap()
        for pair in pairs:
            freq = self.counter[pair]
            if freq <= 0:
                # Invalid frequency for pair
                del self.counter[pair]
            else:
                heapq.heappush(self.heap, PairFreqWrapper((self.counter[pair], pair)))