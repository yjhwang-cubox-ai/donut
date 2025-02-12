from transformers import VisionEncoderDecoderConfig, DonutProcessor, VisionEncoderDecoderModel
from configs.config import config

def init_model_config():
    model_config = VisionEncoderDecoderConfig.from_pretrained(
        config.model_name_or_path, cache_dir=config.cache_dir
    )
    model_config.encoder.image_size = config.image_size
    model_config.decoder.max_length = config.max_length
    return model_config

def init_processor():
    """DonutProcessor 초기화 및 설정"""
    processor = DonutProcessor.from_pretrained(
        config.model_name_or_path, cache_dir=config.cache_dir
    )
    # 이미지 프로세서의 size는 (width, height) 순서여야 함
    processor.image_processor.size = config.image_size[::-1]
    processor.image_processor.do_align_long_axis = False
    return processor

def init_model(model_config, processor):
    """VisionEncoderDecoderModel 초기화 및 토큰 관련 설정"""
    model = VisionEncoderDecoderModel.from_pretrained(
        config.model_name_or_path, config=model_config, cache_dir=config.cache_dir
    )
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s_cord-v2>'])[0]
    return model