from transformers import AutoTokenizer, AutoModel

# pubmedbert_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
#                                                      clean_up_tokenization_spaces=False)
# pubmedbert_model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")


def get_pubmedbert_embedding(text):
    # # Ensure that the tokenizer is handling truncation and padding correctly.
    # inputs = pubmedbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # outputs = pubmedbert_model(**inputs)
    #
    # return outputs.last_hidden_state.mean(dim=1).detach().numpy()
    pass


# scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased",
#                                                   clean_up_tokenization_spaces=False)
# scibert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")


def get_scibert_embedding(text):
    # inputs = scibert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # outputs = scibert_model(**inputs)
    #
    # return outputs.last_hidden_state.mean(dim=1).detach().numpy()
    pass
