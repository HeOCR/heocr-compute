"""CPU inference for the cyttic S44 line-recognition candidate."""

from pathlib import Path


def recognize_line(checkpoint: Path, image: Path) -> str:
    import numpy as np
    import torch
    from PIL import Image
    from tokenizers import Tokenizer
    from transformers import GenerationConfig, VisionEncoderDecoderModel

    # The release has no image processor. Use explicit candidate settings
    # so the input transformation is reproducible and can be reviewed.
    with Image.open(image) as img:
        resized = img.convert("RGB").resize((384, 384), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 127.5 - 1.0

    torch.set_num_threads(1)
    pixels = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    model = VisionEncoderDecoderModel.from_pretrained(
        str(checkpoint), local_files_only=True, use_safetensors=True,
    ).eval()
    generation = GenerationConfig.from_pretrained(str(checkpoint), local_files_only=True)
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    with torch.inference_mode():
        result = model.generate(pixel_values=pixels, generation_config=generation)
    return tokenizer.decode(result[0].tolist(), skip_special_tokens=True) + "\n"
