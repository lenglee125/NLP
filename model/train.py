import torch.optim
from torch import nn
from torch.utils.data import DataLoader

import dataset
from arguments.config import Config
from dataset import AiShellDataset
from dataset_preprocess.pair_word_voice import DataType
from model.model import SpeechTransformer


def train(root='./', save_model="./model.mod"):
    best_loss = float("inf")
    config = Config()
    model = SpeechTransformer()
    train_set = AiShellDataset(DataType.Train, config, root)
    test_set = AiShellDataset(DataType.Test, config, root)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=dataset.pad_collate,num_workers=4)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True, collate_fn=dataset.pad_collate,num_workers=4)

    opt = torch.optim.Adam(model.parameters(), lr=0.000001)

    def one_epoch(epoch=10):
        running_loss = 0
        last_loss = 0
        for i, data in enumerate(train_loader):
            input_v, output_t, il, ol = data

            opt.zero_grad()

            pred = model.forward(input_v, output_t, il, ol)

            loss = nn.functional.cross_entropy(pred,output_t,ignore_index=-1)
            loss.backward()

            opt.step()

            running_loss += loss.item()

            print('  batch {} loss: {}'.format(i + 1, last_loss))

    running_vloss = 0
    for epoch in range(10):
        model.train(True)
        one_epoch(epoch)

        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(test_loader):
                vinputs, vlabels, vlength = vdata
                voutputs = model.forward(vinputs, vlabels, vlength)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)

        print(f"batch {epoch} avg loss: {avg_vloss}")
        if best_loss > avg_vloss:
            best_loss = avg_vloss
            torch.save(model.state_dict(), save_model)
