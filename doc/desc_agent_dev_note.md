## Instance Size and LLM Context Window

If an instance is not very large, we may simply feed the entire instance text to the LLM to generate a description. 

Sampled from some BV benchmarks, below is some smt-lib file size -> token count:

504 kB -> 221k tokens
107 kB -> 54k tokens
52 kB -> 27k tokens
20 kB -> 10k tokens
10 kB -> 5k tokens

Below I list the info about some LLM models we may use.

* The price is per 1M tokens.
| Model | Context Window | Price 
|-------|----------------|-------
| gpt-5-mini | 400k | 0.25 
| o4-mini | 200k | 1.1 
| Claude Sonnet 4.5 | 200k+ | 3 

## Instance Structure Observations

### BV benchmarks

The largest BV benchmark is 113 MB. Among the total 1040 benchmarks, 11 are larger than 1 MB, 32 are larger than 100 kB, and 154 are larger than 10 kB.

Most benchmarks (720/1040) only have one assertion (but maybe a large, long one). 13 benchmarks have more than 10 assertions, with the largest one having 57 assertions.

For other constructs (`declare-fun`, `declare-const`, etc.), only `declare-const` has substantial number in some benchmarks. In particular, there is one benchmark with 715 `declare-const`s. The second largest one has 37 `declare-const`s.
