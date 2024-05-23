import gradio as gr
import numpy as np
import mlflow
import matplotlib.pyplot as plt

model_name = "dcgan_face"
alias = "best_model"
model_uri = f"models:/{model_name}@{alias}"
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(model_uri)

def predict(seed: int):
    """
    Function to generate faces
    Args:
        seed: int = Seed value to generate the random numpy array, which would be
        an input to the generator model of the GAN
    Returns:
        Generated Image
    """
    
    np.random.seed(seed)
    data = np.random.uniform(-1,1, size = (16, 100)).astype("float32")
    samples = loaded_model.predict(data)
    samples = np.expand_dims(samples, axis = 0)

    fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
    count = 0
    for ax, img in zip(axes.flatten(), samples[-1]):
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1)*255 / (2)).astype(np.uint8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((32,32,3)))
        # plt.imsave(str(count) + '.png', img)
        count = count + 1
    fig.savefig("result.png")

    return "result.png"


slider = gr.Slider(0,1000, label = 'Seed Value')


interface = gr.Interface(predict, inputs = slider, outputs = "image", examples= [123,456,789,23])
interface.launch(share = True)