from transformers import VisionEncoderDecoderConfig, DonutProcessor, VisionEncoderDecoderModel
from configs.config import config

def init_model_config():
    model_config = VisionEncoderDecoderConfig.from_pretrained(
        config.model.model_name_or_path, cache_dir=config.model.cache_dir
    )
    model_config.encoder.image_size = config.model.image_size
    model_config.decoder.max_length = config.model.max_length
    return model_config

def init_processor():
    """DonutProcessor 초기화 및 설정"""
    processor = DonutProcessor.from_pretrained(
        config.model.model_name_or_path, cache_dir=config.model.cache_dir
    )
    # 이미지 프로세서의 size는 (width, height) 순서여야 함
    processor.image_processor.size = {'height': config.model.image_size[0], 'width': config.model.image_size[1]}
    # config.model.image_size[::-1]
    processor.image_processor.do_align_long_axis = True # True 가 더 적절하지 않나?(2025.03.04)
    processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<s_ko>", "</s_ko>"]})
    processor.tokenizer.eos_token_id = processor.tokenizer.convert_tokens_to_ids(["</s_ko>"])[0]

    special_tokens = ["<사업자등록번호>", "</사업자등록번호>", "<사업자종류>", "</사업자종류>", "<상호>", "</상호>", "<대표자>", "</대표자>", "<개업연월일>", "</개업연월일>", "<법인등록번호>", "</법인등록번호>", "<사업장소재지>", "</사업장소재지>", "<본점소재지>", "</본점소재지>", "<업태>", "</업태>", "<종목>", "</종목>", "<발급일자>", "</발급일자>", "<발급사유>", "</발급사유>", "<세무서명>", "</세무서명>"]

    processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    return processor

def init_model(model_config, processor):
    """VisionEncoderDecoderModel 초기화 및 토큰 관련 설정"""
    model = VisionEncoderDecoderModel.from_pretrained(
        config.model.model_name_or_path, config=model_config, cache_dir=config.model.cache_dir
    )
    #cord
    # model.config.pad_token_id = processor.tokenizer.pad_token_id
    # model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s_cord-v2>'])[0]
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s_ko>"])[0]
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    return model