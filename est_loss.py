import os
import math
import numpy as np
import torch
from torch.nn import functional as F
from transformer import Model
from buffer import Buffer
import config


def load_data(path, seq_len, train_batch_size, val_batch_size=config.mini_batch_size*2):
    
    train_data = np.load(os.path.join(path, 'train.npy'))
    num_train_seq = (train_data.shape[0]-1) // seq_len
    train_data = train_data[:num_train_seq*seq_len+1]
    train_data = np.stack([train_data[i*seq_len:(i+1)*seq_len+1] for i in range(num_train_seq)], 0)

    train_dataset = Buffer(train_data, train_batch_size)

    valid_data = np.load(os.path.join(path, 'valid.npy'))
    num_seq = (valid_data.shape[0]-1) // seq_len
    valid_data = valid_data[:num_seq*seq_len+1]
    valid_data = np.stack([valid_data[i*seq_len:(i+1)*seq_len+1] for i in range(num_seq)], 0)

    valid_dataset = Buffer(valid_data, val_batch_size)

    print(f'training tokens: {num_train_seq*seq_len/1e9}B')
    print(f'training batches: {len(train_dataset)}')

    # valid_data = torch.from_numpy(valid_data.astype(np.int64))
    # total_seq_len = valid_data.size(0) // valid_batch_size
    # valid_data = valid_data[:total_seq_len*batch_size].reshape(batch_size, total_seq_len)
    # num_batches = math.ceil((total_seq_len - 1) / seq_len)
    # valid_dataset = []
    # for i in range(num_batches):
    #     valid_dataset.append(valid_data[:, i*valid_seq_len : min(total_seq_len, (i+1)*valid_seq_len+1)])

    return train_dataset, valid_dataset

train_dataset, valid_dataset = load_data('./', config.seq_len, config.batch_size)


model = Model(config.num_embeddings, config.input_dim, config.num_heads, config.num_layers, config.tie).cuda()
model.load_state_dict(torch.load('./runs/random_select/best_model.pt'))
model = torch.compile(model)

losses = []


model.eval()
# for data in valid_dataset.seq_sample():
#     data = data.cuda()
#     with torch.inference_mode(), torch.autocast(device_type="cuda"):

#         output = model(data[:, :-1])
#         output = torch.reshape(output, (-1, output.size(-1)))
#         target = torch.reshape(data[:, 1:], (-1,))
#         loss = F.cross_entropy(output, target, reduction='none')

#     losses.append(loss.cpu().numpy())

# avg_ppl = math.exp(np.mean(np.concatenate(losses, 0)).item())
# print(f'valid: {avg_ppl}')

for data in train_dataset.seq_sample(config.mini_batch_size * 2):
    data = data.cuda()

    with torch.inference_mode(), torch.autocast(device_type="cuda"):

        output = model(data[:, :-1])
        output = torch.reshape(output, (-1, output.size(-1)))
        target = torch.reshape(data[:, 1:], (-1,))
        loss = F.cross_entropy(output, target, reduction='none').view(config.mini_batch_size* 2, -1)

    losses.append(loss.cpu().numpy())

losses = np.concatenate(losses, 0)
print(math.exp(losses.mean()))
print(losses.shape)
np.save('./best_loss.npy', losses)