# Readme
This project uses the FastText 300d pre-trained word embeddings downloaded from `https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip`.
The `setup.py` file contains code to download and extract the pre-trained word embeddings to the `pre_trained_embeddings` directory in the current working directory.

# Instructions to run the code
To evaluate the model, run `run_exp.sh` which ends up calling `setup.py` after which it then trains and evaluates the model on both the datasets.

# Report
A brief report detailing the various approaches attempted is located in the `NLP_Assignment_1_Report.pdf` file.