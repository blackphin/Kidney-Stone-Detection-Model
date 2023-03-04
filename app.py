import gradio as gr


def recog_model(input_image):
    return "hey you are sick, and need medical care!!"


demo = gr.Interface(fn=recog_model, inputs=gr.Image(
    type="pil", image_mode="L"), outputs=gr.Label(label="Model Prediction"), allow_flagging="never")

if __name__ == "__main__":
    demo.launch()
