# Collaborative Software Task

## Goal
Build a small, self-contained Python module called `wordstats.py` that exposes a single function `analyze(text: str) -> dict` returning the following keys:

- `word_count` — total number of whitespace-separated words
- `unique_words` — number of unique lowercased words (ignoring punctuation)
- `avg_word_length` — mean length of words (rounded to 2 decimals)
- `top_3` — list of the three most frequent words (lowercased), in descending frequency order; ties broken by alphabetical order

The module must include at least three docstring-tested examples in `analyze`'s docstring and must run correctly with `python -m doctest wordstats.py -v`.

## Constraints
- Pure standard library only (no third-party deps).
- Single file `wordstats.py`.
- The agent team must also produce a 3-line `README.md` summarizing usage.

## Rubric (5 points)
1. `wordstats.py` file exists.
2. `analyze` function is defined and importable.
3. Returns the four expected keys with correct types.
4. `top_3` is correct on the test sentence "the quick brown fox jumps over the lazy dog the fox is quick".
5. README.md file exists with at least 3 non-empty lines.
