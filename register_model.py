import mlflow
import mlflow.pytorch
import torch
from utils import build_network, get_hyperparameters, setup_mlflow
import argparse
from torchinfo import summary
from mlflow import MlflowClient



def register_model(hyperparameters_dict: dict, model_path: str, experiment_name: str):
    """
    Function to register the model in mlflow registry
    Args:
        hyperparameters_dict: dict = dictinary containg hyperparameters
        model_path: str = checkpoint path where the model is stored
        experiment_name: str = name of the experiment for mlflow
    """

    setup_mlflow(experiment_name)
    with mlflow.start_run() as run:
        _, generator = build_network(d_conv_dim=hyperparameters_dict["d_conv_dim"],
                            g_conv_dim=hyperparameters_dict["g_conv_dim"], z_size = hyperparameters_dict["z_size"])
        
        config = torch.load(model_path)
        generator.load_state_dict(config['generator_state_dict'])

        # Log model summary.
        with open("model_summary.txt", "w") as f:
            f.write(str(summary(generator)))
        mlflow.log_artifact("model_summary.txt")
        
        # Log the model
        mlflow.pytorch.log_model(generator,"model",
                                registered_model_name = "generator_face_generation")

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Argument Parser for the model register")
    parser.add_argument("--experiment_name",type = str, 
                        help = "experiment name for the mlflow run", default="/dcgan_v1_exp1")
    parser.add_argument("--hyperparameters_path", type = str, 
                        help = "json file where the hyperparameters are stored", default = "hyperparameters.json")
    parser.add_argument("--checkpoint_path", type = str, 
                        help = "checkpoint path where the GAN state dict is stored", default = "checkpoints/dc_gan.pth")
    
    args = parser.parse_args()

    hyperparameters_dict = get_hyperparameters(args.hyperparameters_path)
    register_model(hyperparameters_dict, args.checkpoint_path, args.experiment_name)