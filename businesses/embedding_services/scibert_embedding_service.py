from transformers import AutoModel, AutoTokenizer

from businesses.embedding_services.embedding_base_service import EmbeddingBaseService
from common.enums.embedding_type import EmbeddingType

embedding_category = EmbeddingType.SciBERT


class SciBertEmbeddingService(EmbeddingBaseService):

    def __init__(self, category: EmbeddingType, max_length: int = None):
        super().__init__(category)

        self.max_length = max_length

        if self.enable_bert_embedding:
            self.scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased",
                                                                   clean_up_tokenization_spaces=False)
            self.scibert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

    def embed(self, text: str) -> (str, bool):

        issue_on_max_length = False

        if self.max_length is None:

            try:
                inputs = self.scibert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                outputs = self.scibert_model(**inputs)
            except Exception as e:
                inputs = self.scibert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                outputs = self.scibert_model(**inputs)
                issue_on_max_length = True
        else:
            inputs = self.scibert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            outputs = self.scibert_model(**inputs)

        return super().parse_string(outputs.last_hidden_state), issue_on_max_length
