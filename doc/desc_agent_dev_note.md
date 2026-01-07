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



