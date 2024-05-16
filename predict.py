import mlflow
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

# helper function for viewing a list of passed in sample images
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
    count = 0
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1)*255 / (2)).astype(np.uint8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((32,32,3)))
        # plt.imsave(str(count) + '.png', img)
        count = count + 1
    fig.savefig("result.png")

mlflow.set_tracking_uri("http://localhost:5000")
device = ("cuda" if torch.cuda.is_available() else "cpu")
logged_model = 'runs:/f8d4efee73f740d59e6da3d16e3e9c0d/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)


data = np.random.uniform(-1,1, size = (16, 100)).astype("float32")
# Predict on a numpy array.
samples = loaded_model.predict(data)
samples = np.expand_dims(samples, axis = 0)

view_samples(-1, samples)