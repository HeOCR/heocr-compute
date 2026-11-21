"""CPU inference for the cyttic S44 line-recognition candidate."""

from pathlib import Path


def line_recognizer(checkpoint: Path):
    import numpy as np
    import torch
    from PIL import Image
    from tokenizers import Tokenizer
    from transformers import GenerationConfig, VisionEncoderDecoderModel

    torch.set_num_threads(1)
    model = VisionEncoderDecoderModel.from_pretrained(
        str(checkpoint), local_files_only=True, use_safetensors=True,
    ).eval()
    generation = GenerationConfig.from_pretrained(str(checkpoint), local_files_only=True)
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    def recognize(image: Path) -> str:
        # The release has no processor. Use explicit candidate settings.
        with Image.open(image) as img:
            resized = img.convert("RGB").resize((384, 384), Image.Resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.float32) / 127.5 - 1.0
        pixels = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        with torch.inference_mode():
            result = model.generate(pixel_values=pixels, generation_config=generation)
        return tokenizer.decode(result[0].tolist(), skip_special_tokens=True)

    return recognize
