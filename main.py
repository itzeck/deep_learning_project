#Based on https://github.com/pytorch/examples/tree/master/dcgan
import torch
import torch.nn as nn
import torch.utils.data
from torch import optim
from torchvision import datasets, transforms
import torchvision.utils
import os

REAL_LABEL = 1
FAKE_LABEL = 0

IMAGE_SIZE = 64


class Generator(nn.Module):
    def __init__(self, latent_dim, width, n_channels):
        super(Generator, self).__init__()
        self.conv1 = nn.ConvTranspose2d(latent_dim, width * 8, 4, 1, 0, bias=False)
        self.batch1 = nn.BatchNorm2d(width * 8)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.ConvTranspose2d(width * 8, width * 4, 4, 2, 1, bias=False)
        self.batch2 = nn.BatchNorm2d(width * 4)
        #nn.ReLU(True)
        self.conv3 = nn.ConvTranspose2d(width * 4, width * 2, 4, 2, 1, bias=False)
        self.batch3 = nn.BatchNorm2d(width * 2)
        #nn.ReLU(True),
        self.conv4 = nn.ConvTranspose2d(width * 2, width, 4, 2, 1, bias=False)
        self.batch4 = nn.BatchNorm2d(width)
        #nn.ReLU(True),
        self.conv5 = nn.ConvTranspose2d(width, width, 4, 2, 1, bias=False)
        self.conv6 = nn.ConvTranspose2d(width, n_channels, 4,2,1, bias=False)
        self.tanh = nn.Tanh()


    def forward(self, x):
        #print(f"Generator in: {x.shape}")
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)
        #print(f"Shape here: {x.shape}")
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu(x)
        #print(f"Shape here: {x.shape}")
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.relu(x)
        #print(f"Shape here: {x.shape}")
        x = self.conv4(x)
        x = self.batch4(x)
        x = self.relu(x)
        #print(f"Shape here: {x.shape}")
        x = self.conv5(x)
        x = self.batch4(x)
        x = self.relu(x)
        x = self.conv6(x)
        #print(f"Shape here: {x.shape}")
        x = self.tanh(x)
        #print("Generator out", x.shape)
        return x


class Discriminator(nn.Module):
    def __init__(self, width, n_channels):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, width, 4, 2, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(width, width * 2, 4, 2, 1, bias=False)
        self.batch1 = nn.BatchNorm2d(width * 2)
        #nn.LeakyReLU(0.2, inplace=True),
        self.conv3 = nn.Conv2d(width * 2, width * 4, 4, 2, 1, bias=False)
        self.batch2 = nn.BatchNorm2d(width * 4)
        #self.leaky_relu = nn.LeakyReLU(0.2, inplace=True),
        self.conv4 = nn.Conv2d(width * 4, width * 8, 4, 2, 1, bias=False)
        self.batch3 = nn.BatchNorm2d(width * 8)
        #nn.LeakyReLU(0.2, inplace=True),
        self.conv5 = nn.Conv2d(width * 8, 1, 4, 1, 0, bias=False)
        self.pool = nn.MaxPool2d(5)
        self.sgm = nn.Sigmoid()


    def forward(self, x):
        #print(f"Discriminator in: {x.shape}")
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.batch1(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.batch2(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.batch3(x)
        x = self.leaky_relu(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = self.sgm(x)
        x = x.view(-1, 1).squeeze(1)
        #print("Discriminator out", x.shape)
        return x


def main(
    batch_size,
    n_epochs,
    learning_rate,
    beta1,
    beta2,
    latent_dim,
    generator_width,
    discriminator_width,
    n_channels,
    out_path,
    image_extension,
):
    # Select GPU if available.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Set seed.
    torch.manual_seed(0)

    # Prepare data.
    transform = transforms.Compose(
        [
            transforms.Resize(2 * IMAGE_SIZE),
            transforms.CenterCrop(2 * IMAGE_SIZE),
            #transforms.CenterCrop(IMAGE_SIZE),
            transforms.RandomAutocontrast(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataloader = torch.utils.data.DataLoader(
        datasets.FashionMNIST("data", download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Prepare batch for visual comparison.
    real_example, _ = next(iter(dataloader))

    grid = torchvision.utils.make_grid(real_example)
    os.makedirs(out_path, exist_ok=True)
    path = os.path.join(out_path, f"real.{image_extension}")
    torchvision.utils.save_image(grid, path, normalize=True)

    # Prepare fixed noise.
    fixed_noise = torch.randn((batch_size, latent_dim, 1, 1), device=device)

    # Prepare generator and discriminator.
    generator = Generator(
        latent_dim=latent_dim, width=generator_width, n_channels=n_channels
    ).to(device)
    generator.load_state_dict(torch.load("./generator.pth", map_location=torch.device('cuda')))
    discriminator = Discriminator(width=discriminator_width, n_channels=n_channels).to(
        device
    )
    discriminator.load_state_dict(torch.load("./discriminator.pth", map_location=torch.device('cuda')))
    

    # Prepare optimizers.
    optimizer_generator = optim.Adam(
        generator.parameters(), lr=learning_rate, betas=(beta1, beta2)
    )
    optimizer_discriminator = optim.Adam(
        discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2)
    )

    # Prepare criterion.
    loss_function = nn.BCELoss()

    # Train.
    generator.train()
    discriminator.train()
    n_batches = len(dataloader)
    for i_epoch in range(n_epochs):
        print(f"Epoch {i_epoch}")
        for i_batch, (batch, _) in enumerate(dataloader):
            # Train discriminator with real data.
            real_data = batch.to(device)
            discriminator.zero_grad()
            expected_real_labels = torch.full(
                (64,), REAL_LABEL, dtype=real_data.dtype, device=device
            )

            prediction = discriminator(real_data)
            #print(expected_real_labels.shape)
            #print(prediction.shape)
            loss_real = loss_function(prediction, expected_real_labels)
            loss_real.backward()

            # Train discriminator on fake data.
            random_noise = torch.randn((64, latent_dim, 1, 1), device=device)
            fake_data = generator(random_noise)
            expected_fake_labels = torch.full(
                (64,), FAKE_LABEL, dtype=real_data.dtype, device=device
            )
            prediction = discriminator(fake_data.detach())
            loss_fake = loss_function(prediction, expected_fake_labels)
            loss_fake.backward()

            optimizer_discriminator.step()

            # Train generator.
            generator.zero_grad()
            expected_false_decision = torch.full(
                (64,), REAL_LABEL, dtype=real_data.dtype, device=device
            )
            prediction = discriminator(fake_data)
            loss_false = loss_function(prediction, expected_false_decision)
            loss_false.backward()
            optimizer_generator.step()

            if i_batch % 10 == 0:
                # Print progress.
                #print(f"{i_epoch = }, {i_batch = }")
                # Process fixed latent.
                generator.eval()
                fake = generator(fixed_noise)

                grid = torchvision.utils.make_grid(fake)
                path = os.path.join(
                    out_path,
                    f"fake-{i_epoch:0{len(str(n_epochs))}}"
                    f"-{i_batch:0{len(str(n_batches))}}"
                    f".{image_extension}",
                )
                torchvision.utils.save_image(grid, path, normalize=True)

    torch.save(generator.state_dict(), "./generator_new.pth")
    torch.save(discriminator.state_dict(), "./discriminator_new.pth")


main(
    batch_size=64,
    n_epochs=25,
    learning_rate=2e-4,
    beta1=0.5,
    beta2=0.999,
    latent_dim=100,
    generator_width=64,
    discriminator_width=64,
    n_channels=1,
    out_path="output",
    image_extension="png",
)
