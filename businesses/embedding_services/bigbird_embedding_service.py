from transformers import BigBirdTokenizer, BigBirdForSequenceClassification

from businesses.embedding_services.embedding_base_service import EmbeddingBaseService
from common.enums.embedding_type import EmbeddingType

embedding_category = EmbeddingType.BigBird_PubMed


class BigBirdEmbeddingService(EmbeddingBaseService):

    def __init__(self, category: EmbeddingType):
        super().__init__(category)

        if self.enable_bert_embedding:
            self.led_tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-large")
            self.led_model = BigBirdForSequenceClassification.from_pretrained("google/bigbird-roberta-large")

    def embed(self, text: str) -> (str, bool):

        inputs = self.led_tokenizer(text, truncation=True, max_length=4096, return_tensors="pt")
        outputs = self.led_model(**inputs)

        return super().parse_string(outputs.last_hidden_state), False
