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
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, width * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(width * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(width * 8, width * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(width * 4, width * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(width * 2, width, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(True),
            nn.ConvTranspose2d(width, n_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, data):
        return self.main(data)


class Discriminator(nn.Module):
    def __init__(self, width, n_channels):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(n_channels, width, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(width, width * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(width * 2, width * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(width * 4, width * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(width * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


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
    discriminator = Discriminator(width=discriminator_width, n_channels=n_channels).to(
        device
    )

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
        for i_batch, (batch, _) in enumerate(dataloader):
            # Train discriminator with real data.
            real_data = batch.to(device)
            discriminator.zero_grad()
            expected_real_labels = torch.full(
                (batch_size,), REAL_LABEL, dtype=real_data.dtype, device=device
            )
            prediction = discriminator(real_data)
            loss_real = loss_function(prediction, expected_real_labels)
            loss_real.backward()

            # Train discriminator on fake data.
            random_noise = torch.randn((batch_size, latent_dim, 1, 1), device=device)
            fake_data = generator(random_noise)
            expected_fake_labels = torch.full(
                (batch_size,), FAKE_LABEL, dtype=real_data.dtype, device=device
            )
            prediction = discriminator(fake_data.detach())
            loss_fake = loss_function(prediction, expected_fake_labels)
            loss_fake.backward()

            optimizer_discriminator.step()

            # Train generator.
            generator.zero_grad()
            expected_false_decision = torch.full(
                (batch_size,), REAL_LABEL, dtype=real_data.dtype, device=device
            )
            prediction = discriminator(fake_data)
            loss_false = loss_function(prediction, expected_false_decision)
            loss_false.backward()
            optimizer_generator.step()

            # Plot/home/itzeck/deeplify/projects/repos/Jetson_Inference
            if i_batch % 10 == 0:
                # Print progress.
                print(f"{i_epoch = }, {i_batch = }")
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

    torch.save(generator.state_dict(), "./generator.pth")
    torch.save(discriminator.state_dict(), "./discriminator.pth")


main(
    batch_size=64,
    n_epochs=50,
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
