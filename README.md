## Face Generation using DCGAN and MLFlow
Pytorch based DCGAN model to generate human faces.

## Install all the required Libraries
```bash
pip install -r requirements.txt
```

## Start the mlflow server using
```bash
mlflow ui
```

## To train the model
```bash
python main.py --experiment_name=dcgan_mlflow --hyperparameters_path=hyperparameters.json --checkpoint_path=checkpoints/dcgag.pth
```

## To download the dataset
The dataset has been preprocessed. Please download the dataset
[click here to download](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be7eb6f_processed-celeba-small/processed-celeba-small.zip)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.