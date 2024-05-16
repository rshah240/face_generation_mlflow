import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from typing import Tuple
import mlflow
import json
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import pickle as pkl


device = ("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.BCEWithLogitsLoss()

def scale(x: torch.Tensor, feature_range: tuple = (-1,1))-> torch.Tensor:
    """ 
    Scale the images in the givenn feature range
    Args:
        x: torch.Tensor = Tensor which is to be scaled
        feature_range: tuple = Tuple which has the range
    Returns:
        torch.Tensor
    """
    min,max = feature_range
    x = x*(max - min) + min
    return x

def setup_mlflow(experiment_name:str):
    """
    Function to set the mlflow witht the
    credentials
    Args:
        experiment_name: str = Name of the experiment for the mlflow
    """

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)


def get_hyperparameters(json_path: str) -> dict:
    """
    Function to get the hyperparameters dictionary
    Args:
        json_path: str = json path to the hyperparameters
    Returns:
        hyperparameters_dict
    """
    with open(json_path, "r") as f:
        hypeparameters_dict = json.load(f)

    return hypeparameters_dict

def real_loss(D_out: torch.Tensor, smooth: bool = True) -> torch.Tensor:
    """
    Loss calculate for the discriminator to classify between
    the output images from the discriminator
    Args:
        D_out: torch.Tensor = Discriminator Output
        smooth: bool = Smoothening bool value
    Returns:
        Loss between true labels and discriminator
    """
    batch_size = D_out.size(0)
    labels = torch.ones(batch_size)

    if smooth:
        labels = labels * 0.9
    labels = labels.to(device)
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out: torch.Tensor) -> torch.Tensor:
    """
    Loss calculate for the discriminator to classify between
    the output images from the discriminator
    Args:
        D_out: torch.Tensor = Discriminator Output
        smooth: bool = Smoothening bool value
    Returns:
        Loss between fake labels and discriminator
    """
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)
    labels = labels.to(device)
    loss = criterion(D_out.squeeze(), labels)
    return loss

def conv(in_channels: int, out_channels: int, kernel_size: int, 
        stride: int = 2, padding: int = 1,
        batch_norm: bool = True) -> nn.Sequential:
    """
        Function that returns the convolution block
    Args:
        in_channels: int = Input Channels
        out_channels: int = Output Channels
        kernel_size: int = Kernel for the convolution layer
        padding: int = Padding for the convolution layer
        batch_norm: bool = if true batchnorm module will be added into
        the sequential layer
    Returns:
        nn.Sequential module
    """
    layers = []
    conv_layers = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                            kernel_size = kernel_size, stride = stride,padding = padding,
                            bias = False)
    layers.append(conv_layers)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


def deconv(in_channels, out_channels, kernel_size, stride =2, padding = 1,
        batch_norm = True):
    """
        Function that returns the deconvolution block
    Args:
        in_channels: int = Input Channels
        out_channels: int = Output Channels
        kernel_size: int = Kernel for the convolution layer
        padding: int = Padding for the convolution layer
        batch_norm: bool = if true batchnorm module will be added into
        the sequential layer
    Returns:
        nn.Sequential module
    """
    layers = []
    conv_layers = nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels,
                                    kernel_size = kernel_size, stride = stride, padding = padding,
                                    bias = False)
    layers.append(conv_layers)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    """
    Class that defines the discriminator block for the
    Generative Adversarial Network
    """
    def __init__(self, conv_dim: int=32):
        """
        Args:
            conv_dim: int = Convolution Dimension whose multiple at every layer
        """
        super(Discriminator, self).__init__()

        # defining convolution layers
        self.conv_dim = conv_dim
        self.conv1 = conv(3, conv_dim, 4,batch_norm=False) # first layer
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2,conv_dim*4,4)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4)

        # final, fully connected layer
        self.fc = nn.Linear(conv_dim*8*2*2,1)

    def forward(self,x):

        out = F.leaky_relu(self.conv1(x))
        out = F.leaky_relu(self.conv2(out))
        out = F.leaky_relu(self.conv3(out))
        out = F.leaky_relu(self.conv4(out))

        out = out.view(-1,self.conv_dim*8*2*2)

        # final output layer
        out = self.fc(out)

        return out


class Generator(nn.Module):
    """
    Class that defines the Generator block for the
    Generative Adversarial Network
    """
    def __init__(self, conv_dim, z_size):
        """
        Args:
            conv_dim: int = Convolution Dimension whose multiple at every layer
            z_size: int = Dimension for the random noise input to generator
        """
        super(Generator, self).__init__()
        self.conv_dim = conv_dim
        self.z_size = z_size

        self.fc = nn.Linear(in_features=z_size, out_features=conv_dim*8*2*2)
        self.t_conv1 = deconv(conv_dim*8, conv_dim*4,4)
        self.t_conv2 = deconv(conv_dim*4, conv_dim*2, 4)
        self.t_conv3 = deconv(conv_dim*2,conv_dim,4)
        self.t_conv4 = deconv(conv_dim,3,4,batch_norm=False)

    def forward(self, x):
        out = F.relu(self.fc(x))
        out = out.view(-1, self.conv_dim*8,2,2)

        out = F.relu(self.t_conv1(out))
        out = F.relu(self.t_conv2(out))
        out = F.relu(self.t_conv3(out))
        out = torch.tanh(self.t_conv4(out))

        return out

def weights_init_normal(m):
    """
    :param m: A module layer in a network
    """
    classname = m.__class__.__name__

    if classname.find('Conv')!= -1:
        torch.nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data,0.0,0.02)


def build_network(d_conv_dim: int, g_conv_dim: int, z_size: int) -> Tuple[nn.Module, nn.Module]:
    """
    Function to build the Generator and Discriminator netwok
    Args:
        d_conv_dim: int = The convolutional dimension for the discriminator
        g_conv_dim: int = The convolutional dimension for the generator
        z_size: int = The input dimension for the uniform distribution
    Returns:
        discriminator, generator
    """
    # define discriminator and generator
    discriminator = Discriminator(d_conv_dim)
    generator = Generator(z_size=z_size, conv_dim=g_conv_dim)

    discriminator.apply(weights_init_normal)
    generator.apply(weights_init_normal)

    discriminator.to(device)
    generator.to(device)

    return discriminator, generator

def get_dataloader(batch_size, image_size, data_dir='processed_celeba_small/') -> DataLoader:
    '''
    Batch the neural network data using DataLoader
    :param batch_size: The size of each batch; the number of images in a batch
    :param image_size:The Square size of the image data (x,y)
    :param data_dir: Directory where image data is located
    :return: DataLoader with batched data
    '''

    image_path = data_dir
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.ToTensor()])

    train_dataset = datasets.ImageFolder(image_path, transform)
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size,
                            shuffle = True, num_workers=8)
    return train_loader


def train(discriminator: nn.Module, generator: nn.Module, train_dataloader: DataLoader,
        hyperparameters_dict: dict, checkpoint_path: str) -> nn.Module:
    """
    Function to train the dataloader
    Args:
        discriminator:nn.Module = Discriminator of the GAN
        generator:nn.Module = Generator of the GAN
        train_dataloader: DataLoader = train dataloader for training of the GAN
        hyperparameters_dict: dict = Dictionary containing hyperparameters
        checkpoint_path: str = Path to store the checkpoint
    """
    
    # Initializing the optimizer
    samples = []
    sample_size = 16
    fixed_z = np.random.uniform(-1,1, size = (sample_size, hyperparameters_dict['z_size']))
    fixed_z = torch.from_numpy(fixed_z).float().to(device)
    d_optimizer = Adam(discriminator.parameters(), lr = hyperparameters_dict['lr'],
                    betas = (hyperparameters_dict['beta_1'], hyperparameters_dict['beta_2']))
    g_optimizer = Adam(generator.parameters(),lr = hyperparameters_dict["lr"],
                    betas = (hyperparameters_dict["beta_1"], hyperparameters_dict["beta_2"]))
    
    n_epochs = hyperparameters_dict['epochs']
    iter_epochs = tqdm(range(n_epochs))


    for epoch in iter_epochs:
        g_losses = 0
        d_losses = 0
        discriminator.train()
        generator.train()
        for idx, (real_images, _) in enumerate(train_dataloader):
            batch_size = real_images.size(0)
            real_images = scale(real_images)

            real_images = real_images.to(device)

            # discriminator training
            d_optimizer.zero_grad()

            out_real = discriminator(real_images)
            d_real_loss = real_loss(out_real)

            z = np.random.uniform(-1,1,size = (batch_size,hyperparameters_dict["z_size"]))
            z = torch.from_numpy(z).float()
            z = z.to(device)
            fake_images = generator(z)
            out_fake = discriminator(fake_images)
            d_fake_loss = fake_loss(out_fake)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # generator training
            g_optimizer.zero_grad()
            z = np.random.uniform(-1,1,size = (batch_size,hyperparameters_dict["z_size"]))
            z = torch.from_numpy(z).float()
            z = z.to(device)

            fake_images = generator(z)
            out_fake_g = discriminator(fake_images)
            g_loss = real_loss(out_fake_g)
            g_loss.backward()
            g_optimizer.step()

            g_losses += g_loss.item()
            d_losses += d_loss.item()

        g_losses = g_losses / len(train_dataloader)
        mlflow.log_metric("Generator_Loss", g_losses, epoch)
        mlflow.log_metric("Discriminator_Loss", d_losses, epoch)
    
        generator.eval()
        with torch.no_grad():
            samples_z = generator(fixed_z)
            samples.append(samples_z)
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    # Saving the model
    config = {}
    config['generator_state_dict'] = generator.state_dict()
    config['discriminator_state_dict'] = discriminator.state_dict()
    config['g_optimizer_state_dict'] = g_optimizer.state_dict()
    config['d_optimizer_state_dict'] = d_optimizer.state_dict()
    torch.save(config, checkpoint_path)

    return generator

        






