"""
author:- Rachit Shah
Main file to train the DCGAN
"""
from utils import get_dataloader, build_network, setup_mlflow, \
    train, get_hyperparameters, criterion
import argparse
from torchinfo import summary
import mlflow


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Arguments for the DCGAN")
    parser.add_argument("--experiment_name",type = str, 
                        help = "experiment name for the mlflow run", default="/dcgan_v1_exp1")
    parser.add_argument("--hyperparameters_path", type = str, 
                        help = "json file where the hyperparameters are stored", default = "hyperparameters.json")
    parser.add_argument("--checkpoint_path", type = str, 
                        help = "checkpoint path where the GAN state dict will be stored", default = "checkpoints/dc_gan.pth")
    
    
    args = parser.parse_args()
    setup_mlflow(args.experiment_name)
    with mlflow.start_run() as run:
        # load the hyperparameters json file
        hyperparameters_dict = get_hyperparameters(args.hyperparameters_path)
        # mlflow to log hyperparameters
        mlflow.log_params(hyperparameters_dict)
        train_dataloader = get_dataloader(batch_size = hyperparameters_dict["batch_size"], image_size = 32)
        discriminator, generator = build_network(d_conv_dim=hyperparameters_dict["d_conv_dim"],
                                                g_conv_dim=hyperparameters_dict["g_conv_dim"], z_size = hyperparameters_dict["z_size"])
        generator = train(discriminator, generator, train_dataloader, hyperparameters_dict, args.checkpoint_path)

        # Log model summary.
        with open("model_summary.txt", "w") as f:
            f.write(str(summary(generator)))
        mlflow.log_artifact("model_summary.txt")
        
        mlflow.pytorch.log_model(generator,"model")






