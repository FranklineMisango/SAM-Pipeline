from PIL import Image
from lang_sam import LangSAM

model = LangSAM()
image_pil = Image.open("Test.jpg").convert("RGB")
text_prompt = "car"
masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)