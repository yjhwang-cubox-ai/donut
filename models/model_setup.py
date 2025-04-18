from transformers import VisionEncoderDecoderConfig, DonutProcessor, VisionEncoderDecoderModel
from configs.config import config

def init_processor():
    # 프로세서 로드
    processor = DonutProcessor.from_pretrained(
        config.model.model_name_or_path, cache_dir=config.model.cache_dir
    )
    processor.image_processor.size = {'height': config.model.image_size[0], 'width': config.model.image_size[1]}
    processor.image_processor.do_align_long_axis = True
    processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<s_ko>", "</s_ko>"]})
    processor.tokenizer.eos_token_id = processor.tokenizer.convert_tokens_to_ids(["</s_ko>"])[0]
    
    special_tokens = ["<HEAD>", "</HEAD>", "<사업자종류>", "</사업자종류>", "<사업자등록번호>", "</사업자등록번호>", "<상호>", "</상호>", "<대표자>", "</대표자>", "<개업연월일>", "</개업연월일>", "<법인등록번호>", "</법인등록번호>", "<사업장소재지>", "</사업장소재지>", "<본점소재지>", "</본점소재지>", "<업태>", "</업태>", "<종목>", "</종목>", "<발급사유>", "</발급사유>", "<발급일자>", "</발급일자>", "<세무서명>", "</세무서명>"]
    processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    return processor

def init_model_and_processor():
    # 설정 로드
    model_config = VisionEncoderDecoderConfig.from_pretrained(
        config.model.model_name_or_path, cache_dir=config.model.cache_dir
    )
    model_config.encoder.image_size = config.model.image_size
    model_config.decoder.max_length = config.model.max_length   
    
    # 프로세서 로드
    processor = init_processor()
    
    # 모델 로드
    model = VisionEncoderDecoderModel.from_pretrained(
        config.model.model_name_or_path,
        config=model_config,
        cache_dir=config.model.cache_dir
    )
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s_ko>"])[0]
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    
    return model, processor