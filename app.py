import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import tensorflow as tf
import cv2

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_model.keras')

model = load_model()

st.title("MNIST Digit Recognizer")
st.write("Draw")

from streamlit_drawable_canvas import st_canvas

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.0)",
    stroke_width=32,          # thicker = better for 6/7 distinction
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=340,
    width=340,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        img = ImageOps.invert(img)
        img = img.filter(ImageFilter.SMOOTH_MORE)   # extra smoothing

        img_array = np.array(img).astype('float32')

        # Stronger preprocessing
        _, thresh = cv2.threshold(img_array, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            
            cropped = img_array[y:y+h, x:x+w]
            
            # Target size slightly larger for better loop detection in 6
            scale = 22.0 / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            final_img = np.zeros((28, 28), dtype='float32')
            pad_x = (28 - new_w) // 2
            pad_y = (28 - new_h) // 2
            final_img[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        else:
            final_img = np.zeros((28, 28), dtype='float32')

        final_img = final_img / 255.0
        input_img = np.expand_dims(final_img, axis=0)

        logits = model.predict(input_img, verbose=0)
        probs = tf.nn.softmax(logits[0]).numpy()
        pred = int(np.argmax(probs))
        conf = float(probs[pred] * 100)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(final_img, caption="Processed 28×28", width=220, clamp=True)
        
        with col2:
            st.success(f"**Predicted: {pred}**")
    else:
        st.warning("Draw something first!")

 
