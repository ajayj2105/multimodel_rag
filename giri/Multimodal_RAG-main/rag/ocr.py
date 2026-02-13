import easyocr
import os
import tempfile


def extract_text_from_image(image_path):
    reader = easyocr.Reader(["en"])
    result = reader.readtext(image_path)

    text = " ".join([t for (_, t, _) in result])
    return text.strip()


def extract_text_from_image_bytes(image_bytes, suffix=".jpg"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(image_bytes)
        temp_path = temp_file.name

    try:
        return extract_text_from_image(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
