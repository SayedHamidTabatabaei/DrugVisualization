from transformers import AutoTokenizer, AutoModel

from businesses.embedding_services.embedding_base_service import EmbeddingBaseService
from common.enums.embedding_type import EmbeddingType

embedding_category = EmbeddingType.PubMedBERT


class PubmedBertEmbeddingService(EmbeddingBaseService):

    def __init__(self, category: EmbeddingType):
        super().__init__(category)
        if self.enable_bert_embedding:
            self.pubmedbert_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                                                                      clean_up_tokenization_spaces=False)
            self.pubmedbert_model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

    def embed(self, text: str) -> (str, bool):
        issue_on_max_length = False

        try:
            inputs = self.pubmedbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.pubmedbert_model(**inputs)
        except Exception as e:
            inputs = self.pubmedbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = self.pubmedbert_model(**inputs)
            issue_on_max_length = True
            # print(f'Exception message {e}')

        return EmbeddingBaseService.parse_string(outputs.last_hidden_state), issue_on_max_length
