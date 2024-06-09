import streamlit as st
from lang_sam import LangSAM
import numpy as np
from PIL import Image, ImageDraw, ImageFont



#page config
st.set_page_config(layout="wide")

# Initialize the model
model = LangSAM()

# Streamlit app title and instructions
st.title("SAM Multilabelling App")
st.write("Import a picture below")

# File uploader for the image
original_image = st.file_uploader("Drop the image", type=['png', 'jpg', 'jpeg'])

if original_image:
    st.success("Image captured")
    image_pil = Image.open(original_image).convert("RGB")

    # Number of objects the user wants to label
    num_objects = st.number_input("How many things in the image would you like to label?", min_value=1, step=1)
    if num_objects:
        st.success(f"Number of objects you want to label are confirmed as {num_objects}")

        # Collecting text prompts from the user
        text_prompts = []
        for i in range(num_objects):
            text_prompt = st.text_input(f"Enter the thing you want to label (Label {i+1})")
            if text_prompt:
                text_prompts.append(text_prompt)

        if len(text_prompts) == num_objects and st.button("Label!"):
            try:
                # Perform predictions for each text prompt
                predictions = []
                for text_prompt in text_prompts:
                    # Ensure the predict method is called correctly without unexpected arguments
                    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
                    predictions.append((masks, boxes, phrases, logits))

                # Draw predictions for each text prompt on the input image
                image_array = np.asarray(image_pil)
                draw = ImageDraw.Draw(image_pil)
                font = ImageFont.load_default()  # You can choose a custom font if needed

                # Define a list of colors for labeling
                colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta"]

                for i, (masks, boxes, phrases, logits) in enumerate(predictions):
                    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
                    label_color = colors[i % len(colors)]  # Change label color based on the prompt

                    for label, box in zip(labels, boxes):
                        x0, y0, x1, y1 = box  # Unpack the coordinates
                        draw.rectangle([x0, y0, x1, y1], outline=label_color)  # Draw the rectangle
                        draw.text((x0, y0), label, fill=label_color, font=font)  # Draw the label

                # Display the output image
                st.image(image_pil, caption="Labeled Image")
            except TypeError as e:
                st.error(f"TypeError in model.predict: {e}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
else:
    st.warning("Please upload an image to proceed.")
