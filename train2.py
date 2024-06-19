# %%
import os
import math
import numpy as np
import torch
from torch.nn import functional as F
from transformer import Model
from buffer import Buffer
import config
import transformer

path = './'


model_name = 'trans'

class DataSet(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, seq_len):
        super().__init__()

        self.num_samples = (data.shape[0]-1) // seq_len

        data = data[:self.num_samples*seq_len+1]

        self.data = [data[i*seq_len : (i+1)*seq_len+1] for i in range(self.num_samples)]

    def __getitem__(self, i):
        return torch.from_numpy(self.data[i].astype(np.int64))
    
    def __len__(self):
        return self.num_samples
    
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


# %%


train_dataset, valid_dataset = load_data(path, config.seq_len, config.batch_size)

# train_tokens = len(train_dataset) * seq_len
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
# valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=mini_batch_size, pin_memory=True, num_workers=4)
# test_dataset = DataSet(dict['test'], 10, 512)
# %%
import math



# %%
# import transformers
# transformers.GPT2LMHeadModel
original_model = Model(config.num_embeddings, config.input_dim, config.num_heads, config.num_layers, config.tie).cuda()
model = torch.compile(original_model)
    # model = origin_model
num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
print(f'number of param: {num_params}')

from tqdm import tqdm


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for n, p in module.named_parameters():
        if 'norm' in n or 'bias' in n:
            group_no_decay.append(p)
        else:
            group_decay.append(p)
        
    print(len(group_no_decay))
    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups



weights = group_weight(model)
optimizer = torch.optim.AdamW(weights, lr=config.max_lr, weight_decay=config.wd, betas=(0.9, 0.98))
# optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=wd, betas=(0.9, 0.98))

scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=config.warmup_steps)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_dataset.num_batch*config.train_epoches-config.warmup_steps, eta_min=config.max_lr*0.01)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], [config.warmup_steps])

# %%
import math

from torch.utils.tensorboard import SummaryWriter

test_name = 'drop_sequence_5'
writer = SummaryWriter(path+'/runs/'+test_name, max_queue=120)

print(model)
scaler = torch.cuda.amp.GradScaler()

train_step = 0
best_loss = float('inf')
losses = []

for i in range(config.train_epoches):
    writer.add_scalar('Stat/drop_sequence', len(train_dataset.drop_idx), i)
    for data, data_idxes in train_dataset.sample():
        data = data.cuda()

        assert config.batch_size % config.mini_batch_size == 0
        num_mini_batches = config.batch_size // config.mini_batch_size
        batch_loss = 0
        model.train()

        mask_list = []
        for mini_data, mini_idx in zip(data.chunk(num_mini_batches, 0), data_idxes.chunk(num_mini_batches, 0)):
            with torch.autocast(device_type="cuda"):
                output = model(mini_data[:, :-1])
                final_output = torch.reshape(output, (-1, output.shape[-1]))
                final_target = torch.reshape(mini_data[:, 1:], (-1,))
                token_loss = F.cross_entropy(final_output, final_target, reduction='none').view(mini_data.size(0), -1)
                # token_loss_list.append(token_loss.detach().clone().cpu().numpy())
                mask = train_dataset.update(token_loss.detach().clone().cpu().numpy(), mini_idx.numpy(), train_step)
                # mask_list.append(mask)
                # if i >= 1:
                #     mask = mask.cuda()
                #     token_loss = torch.masked_select(token_loss, mask) 
                loss = token_loss.mean() / num_mini_batches

            batch_loss += loss.item()

            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        # token_loss_list = np.concatenate(token_loss_list, 0)

        # train_dataset.update(token_loss_list, data_idxes, train_step)

        writer.add_scalar('Loss/train', math.exp(batch_loss), train_step)


        losses.append(batch_loss)

        train_step += 1


    avg_loss = math.exp(sum(losses)/len(losses))
    writer.add_scalar('Loss/train_average', avg_loss, train_step)
    
    losses.clear()
    model.eval()
    for data in valid_dataset.seq_sample():
        data = data.cuda()
        with torch.inference_mode(), torch.autocast(device_type="cuda"):
            
            output = model(data[:, :-1])
            final_output = torch.reshape(output, (-1, output.shape[-1]))
            final_target = torch.reshape(data[:, 1:], (-1,))
            loss = F.cross_entropy(final_output, final_target, reduction='none')

        losses.append(loss.cpu().numpy())
    
    avg_loss = math.exp(np.mean(np.concatenate(losses, 0)).item())
    print(f'\n valid {train_step}: {avg_loss}')

    writer.add_scalar('Loss/valid', avg_loss, train_step)

    losses.clear()



    if avg_loss <= best_loss:
        print(avg_loss)
        best_loss = avg_loss
        torch.save(original_model.state_dict(), path+'/runs/'+test_name+'/best_model.pt')

    model.train()
    # np.save(f'{i+1}.npy', train_dataset.grad)

# losses.clear()
# model.eval()
# for data in valid_dataset.seq_sample():
#     data = data.cuda()
#     with torch.inference_mode(), torch.autocast(device_type="cuda"):
        
#         output = model(data[:, :-1])
#         final_output = torch.reshape(output, (-1, output.shape[-1]))
#         final_target = torch.reshape(data[:, 1:], (-1,))
#         loss = F.cross_entropy(final_output, final_target, reduction='none')

#     losses.append(loss.cpu().numpy())

# avg_loss = math.exp(np.mean(np.concatenate(losses, 0)).item())
# print(f'valid {train_step}: {avg_loss}')

# writer.add_scalar('Loss/valid', avg_loss, train_step)

# losses.clear()

# if avg_loss <= best_loss:
#     print(avg_loss)
#     best_loss = avg_loss
#     torch.save(origin_model.state_dict(), path+'/runs/'+test_name+'/best_model.pt')

