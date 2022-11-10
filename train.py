import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


def train(dataset, test_noise=test_noise, epochs=epochs, noise_size=gf, batch_size=batch_size, linear=True, learning_rate=0.0002, beta=0.9):

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_batch = len(loader)

    def img_noise(size):
        if linear:
            n = torch.randn(size, noise_size)
        else:
            n = torch.randn(size, noise_size, 1, 1)
        return n

    images = []
    g_loss = []
    d_loss = []

    for epoch in range(epochs):

        print('-'*40)

        for i, (real_data, _) in enumerate(loader):

            N = real_data.shape[0]

            optimD.zero_grad()

            # Discriminator training on real data
            real_data = real_data.to(device)
            labels = torch.ones((N, 1), dtype=torch.float, device=device)

            output = dNet(real_data)
            dx_real = output.mean().item()
            error_dreal = criterion(output, labels)
            error_dreal.backward()

            # Discriminator training on fake data
            # here the noise will be used also for generator.
            noise = img_noise(N)
            fake_data = gNet(noise.to(device)).detach()
            labels = torch.zeros((N, 1), dtype=torch.float, device=device)

            output = dNet(fake_data)
            dx_fake = output.mean().item()
            error_dfake = criterion(output, labels)
            error_dfake.backward()

            errorD = error_dreal + error_dfake
            optimD.step()

            # Generator training
            optimG.zero_grad()
            labels = torch.ones((N, 1), dtype=torch.float, device=device)

            fake_data = gNet(noise.to(device))
            output = dNet(fake_data)
            dx_gen = output.mean().item()
            errorG = criterion(output, labels)
            errorG.backward()
            optimG.step()

            g_loss.append(errorG.item())
            d_loss.append(errorD.item())
            if (i) % 100 == 0:
                print(f'Epoch: [{epoch+1}/{epochs}] Batch: [{i}/{num_batch}]')
                print(f'Loss_D: {errorD.item()} Loss_G: {errorG.item()}')
                print(f'D(x): {dx_real} D(G(z)): {dx_fake}')

                with torch.no_grad():
                    fake_imgs = gNet(test_noise.to(device)).detach().cpu()
                    if linear:
                        fake_imgs = fake_imgs.view(-1, 1, df, df)
                    fig, ax = plt.subplots(figsize=(16, 16))
                    grid = make_grid(
                        fake_imgs, normalize=True, scale_each=True)
                    images.append(grid)
                    h_grid = make_grid(
                        fake_imgs[:16], nrows=10, normalize=True, scale_each=True)
                    ax = ax.imshow(np.transpose(h_grid, (1, 2, 0)))
                    plt.axis('off')
                    plt.show()

    return images, g_loss, d_loss
