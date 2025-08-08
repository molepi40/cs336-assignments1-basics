from functools import lru_cache
@lru_cache
def bytes_to_unicode() -> dict[int, bytes]:
    unprintable_ranges = [
        (0, ord("!")),
        (ord("~") + 1, ord("¡")),
        (ord("¬") + 1, ord("®")),
        (ord("ÿ") + 1, 256)
    ]
    bytes_list = range(256)
    unicode_list = list(range(256))
    offset = 256
    for unprintable_range in unprintable_ranges:
        for i in range(unprintable_range[0], unprintable_range[1]):
            unicode_list[i] = offset + i
    
    bytes_to_unicode = dict(
        zip(
            bytes_list, 
            map(chr, unicode_list)
        )
    )
    return bytes_to_unicode

if __name__ == '__main__':
    # Test function
    bytes_to_unicode_dict = bytes_to_unicode()
    for key, val in bytes_to_unicode_dict.items():
        print(str(key) + " " + val)