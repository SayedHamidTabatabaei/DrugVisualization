from transformers import LongformerTokenizer, LongformerForSequenceClassification

from businesses.embedding_services.embedding_base_service import EmbeddingBaseService
from common.enums.embedding_type import EmbeddingType

embedding_category = EmbeddingType.LongFormer_BioNER


class LongFormerEmbeddingService(EmbeddingBaseService):

    def __init__(self, category: EmbeddingType):
        super().__init__(category)

        if self.enable_bert_embedding:
            self.longformer_tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
            self.longformer_model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096")

    def embed(self, text: str) -> (str, bool):

        inputs = self.longformer_tokenizer(text, truncation=True, max_length=4096, return_tensors="pt")
        outputs = self.longformer_model(**inputs, output_hidden_states=True)

        return super().parse_string(outputs.hidden_states[-1]), False
