from transformers import LEDForSequenceClassification, LEDTokenizer

from businesses.embedding_services.embedding_base_service import EmbeddingBaseService
from common.enums.embedding_type import EmbeddingType

embedding_category = EmbeddingType.LED


class LedEmbeddingService(EmbeddingBaseService):

    def __init__(self, category: EmbeddingType):
        super().__init__(category)

        if self.enable_bert_embedding:
            self.led_tokenizer = LEDTokenizer.from_pretrained("allenai/led-large-16384")
            self.led_model = LEDForSequenceClassification.from_pretrained("allenai/led-large-16384")

    def embed(self, text: str) -> (str, bool):

        inputs = self.led_tokenizer(text, truncation=True, max_length=16384, return_tensors="pt")
        outputs = self.led_model(**inputs)

        return super().parse_string(outputs.encoder_last_hidden_state), False
