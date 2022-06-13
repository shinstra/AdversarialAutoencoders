"""

"""
import os

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
import torchvision
from torchvision.transforms import PILToTensor

from backend import *

# ---------- #
# Parameters #
# ---------- #

CACHE_PATH = os.path.join(os.path.split(__file__)[0], 'cache')
MODEL = AutoEncoderMLP

EPOCHS = 20
BATCHSIZE = 32
LOGGING_FREQ = 10

# -------------- #
# Helper Methods #
# -------------- #

def preproc_mnist(batch):
    images, labels = list(zip(*batch))
    images = torch.cat([t.flatten()[None, :] for t in images], dim=0) / 255.0
    labels = torch.cat([one_hot(torch.tensor(v), 10)[None, :] for v in labels], dim=0)
    return images, labels

# ------ #
# Script #
# ------ #

def main():
    """"""

    # Setup Cache
    if not os.path.exists(CACHE_PATH):
        os.mkdir(CACHE_PATH)
    save_folder = generate_cache_folder(CACHE_PATH, MODEL.__name__)

    # Prepare Dataset (MNIST)
    train_ds = torchvision.datasets.MNIST(root=CACHE_PATH, train=True, transform=PILToTensor(), download=True)
    valid_ds = torchvision.datasets.MNIST(root=CACHE_PATH, train=False, transform=PILToTensor(), download=True)

    # Prepare Dataset Pipelines
    train_dl = DataLoader(train_ds, batch_size=BATCHSIZE, shuffle=True, num_workers=4, collate_fn=preproc_mnist)
    valid_dl = DataLoader(valid_ds, batch_size=BATCHSIZE, shuffle=True, num_workers=4, collate_fn=preproc_mnist)
    valid_il = iter(valid_dl)

    # Prepare Model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = AutoEncoderMLP(
        input_size=784,
        enc_hidden_size=[1000, 1000],
        latent_size=10,
        dec_hidden_size=[1000, 1000],
        output_size=784,
        activation_fn=torch.nn.ReLU,
        keep_prob=0.8
    )
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=0.0005)
    loss_fn = torch.nn.MSELoss()

    train_loss = []
    valid_loss = []
    log_steps = []
    step = 0
    for epoch in range(EPOCHS):
        for b, (batch, _) in enumerate(train_dl):

            batch = batch.to(device)
            outputs = model(batch)
            loss = loss_fn(batch, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step % LOGGING_FREQ) == 0:
                with torch.no_grad():
                    # Pull valid batch
                    try:
                        vbatch, _ = next(valid_il)
                    except StopIteration:
                        # Need to reset iterator
                        valid_il = iter(valid_dl)
                        vbatch, _ = next(valid_il)

                    vbatch = vbatch.to(device)
                    vloss = loss_fn(model(vbatch), vbatch)

                print(f'Epoch: {epoch}, Batch: {b}, Step: {step} >> train: {loss.item():0.4f}, valid: {vloss.item():0.4f}')

            step += 1









            pass


    pass




if __name__ == '__main__':
    import argparse
    main()
