1. What is "self-supervised learning"?
1. What is a "language model"?
1. Why is a language model considered self-supervised?
1. What are self-supervised models usually used for?
1. Why do we fine-tune language models?
1. What are the three steps to create a state-of-the-art text classifier?
1. How do the 50,000 unlabeled movie reviews help us create a better text classifier for the IMDb dataset?
1. What are the three steps to prepare your data for a language model?
1. What is "tokenization"? Why do we need it?
1. Name three different approaches to tokenization.
1. What is `xxbos`?
1. List four rules that fastai applies to text during tokenization.
1. Why are repeated characters replaced with a token showing the number of repetitions and the character that's repeated?
1. What is "numericalization"?
1. Why might there be words that are replaced with the "unknown word" token?
1. With a batch size of 64, the first row of the tensor representing the first batch contains the first 64 tokens for the dataset. What does the second row of that tensor contain? What does the first row of the second batch contain? (Careful—students often get this one wrong! Be sure to check your answer on the book's website.)
1. Why do we need padding for text classification? Why don't we need it for language modeling?
1. What does an embedding matrix for NLP contain? What is its shape?
1. What is "perplexity"?
1. Why do we have to pass the vocabulary of the language model to the classifier data block?
1. What is "gradual unfreezing"?
1. Why is text generation always likely to be ahead of automatic identification of machine-generated texts?