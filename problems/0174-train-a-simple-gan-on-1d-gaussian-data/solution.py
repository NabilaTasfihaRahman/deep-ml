import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
def train_gan(mean_real: float, std_real: float, latent_dim: int = 1, hidden_dim: int = 16, learning_rate: float = 0.001, epochs: int = 5000, batch_size: int = 128, seed: int = 42):
    torch.manual_seed(seed)
    # Your PyTorch implementation here
    class Gan(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent_dim=latent_dim
            self.hidden_dim=hidden_dim
            self.gen_layer=nn.Sequential(nn.Linear(self.latent_dim,
            self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim,1))
            
            self.dis_layer=nn.Sequential(nn.Linear(1,self.hidden_dim),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_dim,1),
                                                nn.Sigmoid())
        def forward(self,z):
            self.gen_output=self.gen_layer(z)
            self.dis_output=self.dis_layer(self.gen_output)
            
            return self.gen_output,self.dis_output
    model=Gan()
    gen_optimizer=torch.optim.SGD(model.gen_layer.parameters(),lr=learning_rate)
    dis_optimizer=torch.optim.SGD(model.dis_layer.parameters(),lr=learning_rate)
    for i in range(epochs):
        real_samples=torch.normal(mean=mean_real,std=std_real,size=(batch_size,1))
        noise=torch.normal(mean=0.0,std=1.0,size=(batch_size,latent_dim))
        real_labels=torch.ones((batch_size,1))
        fake_labels=torch.zeros((batch_size,1))
        real_prediction=model.dis_layer(real_samples)
        fake_samples=model.gen_layer(noise)

        dis_optimizer.zero_grad()
        fake_prediction=model.dis_layer(fake_samples.detach())
        real_loss= F.binary_cross_entropy(real_prediction,real_labels)                    
        fake_loss=F.binary_cross_entropy(fake_prediction, fake_labels)
        dis_loss=real_loss+fake_loss
        dis_loss.backward()
        dis_optimizer.step()

        gen_optimizer.zero_grad()
        fake_prediction=model.dis_layer(fake_samples)
        gen_loss=F.binary_cross_entropy(fake_prediction,real_labels)
        gen_loss.backward()
        gen_optimizer.step()
    
    def gen_forward(z):
        with torch.no_grad():
            return model.gen_layer(z)
    return gen_forward

