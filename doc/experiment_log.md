## LLM Generated Descriptions

I have used the `gpt-oss-120b` model to generate descriptions for BV benchmarks. Basically, the prompt includes instructions, meta info about the SMT-LIB file, and the SMT-LIB file content (now truncated to the first 200k characters). Now I have created two versions of descriptions:

*Version 0*: ask the LLM to generate descriptions in 4–6 sentences. You can check the prompt [here](https://github.com/JohnLyu2/SMT_AS_LM/blob/3e1d071208336f47039e7b81bea0ac4cacb135de/src/prompt.py#L222). The generated descriptions are saved in this [JSON file](../data/llm_descs/v0/BV.json).
The generated descriptions has a median length of 342 tokens (min: 214, 25%: 311, 75%: 380, max: 709). 

For `all-MiniLM-L6-v2` (max token length: 256), 97.6% of descriptions are truncated; for `all-mpnet-base-v2` (max token length: 384), 22.5% of descriptions are truncated.

*Version 1*: to generate shorter descriptions to better fit the context window of smaller models such as `all-MiniLM-L6-v2` having a max token length of 256, I've created a second version of the prompt [here](https://github.com/JohnLyu2/SMT_AS_LM/blob/d471c7f2214426fc401376d2eed4f7a617d091a6/src/prompt.py#L222). The generated descriptions are saved in this [JSON file](../data/llm_descs/v1/BV.json). The generated descriptions has a median length of 189 tokens (min: 119, 25%: 172, 75%: 211, max: 429). 

## Algorithm Selection Results: LLM-generated Descriptions + Pretrained Embedding Models

|  | Train | Test |
| --- | --- | --- |
| synt | 95.3 | 32.0 |
| native desc (all-mpnet-base-v2) | 39.0 | 7.4 |
| synt + native desc (all-mpnet-base-v2) | 81.7 | 31.2 |
| LLM desc v0 (all-MiniLM-L6-v2) | 100.0 | 14.7 |
| LLM desc v0 (all-MiniLM-L6-v2) - parameter-tuned SVM | 100.0 | 21.0 |
| synt + LLM desc v0 (all-MiniLM-L6-v2) - parameter-tuned SVM | 100.0 | 17.1 |
| LLM desc v0 (all-mpnet-base-v2) | 100.0 | -3.9 |
| LLM desc v1 (all-MiniLM-L6-v2) | 100.0 | -9.4 |

One key observation of LLM-generated descriptions combined with pretrained embedding models is a strong tendency to overfit. Since the LLM-generated descriptions are all different from each other, the pretrained embedding models map them into a high-dimensional space. After training, the algorithm selector can always achieve perfect performance on the training set, but the performance on the test set is usually not good. My intepreation is that now training is like memorizing the training set as "which specific description corresponds to which solver", but it does not generalize to unseen descriptions.

I tried various approaches, e.g., PCA, different ML models, the best I got is fining-tuning the C parameter in SVM (listed above). Although, in some combinations, the test results can be better than native descriptions,they do not provide much complementary information to syntactic features, since the training performance without syntactic features is already perfect.

## Algorithm Selection Results: Finetuning Embedding Models

One direction that may be promising is to fine-tune the embedding models. Pretrained embedding models are trained on large text corpora for general-purpose language understanding tasks. Fine-tuning these models for our task would encourage them to generate embeddings that are closer for descriptions whose selected solvers are the same or similar, and farther apart for descriptions whose selected solvers differ. This task-aware embedding space may mitigate the overfitting issue, as it can guide the models to focus on patterns in the descriptions that are more informative for distinguishing solver behaviors.

I currently implemented the fine-tuning pipeline based on [SetFit](https://github.com/huggingface/setfit). I tried the simplest fine-tuning task as "classifying the best solver used for each description". Using the fine-tuned embedding models, I observed the the overfitting is mitigated. I am working on trying different fine-tuning tasks, embedding models, and discription combinations. Will updatet the results once collected more data.
