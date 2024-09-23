from transformers import AutoModel, AutoTokenizer

from businesses.embedding_services.embedding_base_service import EmbeddingBaseService
from common.enums.embedding_type import EmbeddingType

embedding_category = EmbeddingType.SciBERT


class SciBertEmbeddingService(EmbeddingBaseService):

    def __init__(self, category: EmbeddingType):
        super().__init__(category)
        if self.enable_bert_embedding:
            self.scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased",
                                                                   clean_up_tokenization_spaces=False)
            self.scibert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

    def embed(self, text: str) -> (str, bool):

        issue_on_max_length = False

        try:
            inputs = self.scibert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.scibert_model(**inputs)
        except Exception as e:
            inputs = self.scibert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = self.scibert_model(**inputs)
            issue_on_max_length = True
            print(f'Exception message {e}')

        return EmbeddingBaseService.parse_string(outputs.last_hidden_state.mean(dim=1).detach().numpy()), issue_on_max_length
