import numpy as np
import scipy.special
import torch
import scipy
import math
# save loss
# class Buffer:
#     def __init__(self, data, batch_size) -> None:
        
#         seq_len = data.shape[1]-1

#         self.batch_size = batch_size
#         self.num_batch = data.shape[0] // batch_size
 
#         self.data = data[:self.batch_size*self.num_batch]

#         self.size = self.data.shape[0]

#         self.update_step = np.zeros(self.size, dtype=np.uint32)
#         self.loss = np.zeros((self.size, seq_len), dtype=np.float16)

#         self.save_idx = np.random.choice(self.size, 1024, False)

#     def update(self, loss: np.ndarray, idx: np.ndarray, update_step: int):


#         self.update_step[idx] = update_step
#         self.loss[idx] = loss
    
#     def sample(self):

#         idxes = np.random.choice(self.size, self.size, replace=False)

#         for i in range(self.num_batch):
#             batch_idxes = idxes[i*self.batch_size:(i+1)*self.batch_size]
#             yield torch.from_numpy(self.data[batch_idxes].astype(np.int64)), batch_idxes

#     def __len__(self):
#         return self.num_batch
    
#     def seq_sample(self):
#         num_batches = self.size // self.batch_size

#         for i in range(num_batches):
#             idxes = np.arange(i*self.batch_size,(i+1)*self.batch_size)
#             yield torch.from_numpy(self.data[idxes].astype(np.int64))

#     def save(self, i):

#         np.save(f'./{i}_idx.npy', self.update_step[self.save_idx])
#         np.save(f'./{i}_loss.npy', self.loss[self.save_idx])

    

# class Buffer:
#     def __init__(self, data, batch_size) -> None:
        
#         seq_len = data.shape[1]-1

#         self.batch_size = batch_size
#         self.num_batch = data.shape[0] // batch_size 
 
#         self.data = data[:self.batch_size*self.num_batch]

#         self.size = self.data.shape[0]

#         self.last_step = np.zeros(self.size, dtype=np.uint32)
#         self.loss = np.zeros((self.size, seq_len), dtype=np.float16)
#         self.grad = np.zeros((self.size, seq_len), dtype=np.float16)

#         self.decay = 0.9998

#     def update(self, loss: np.ndarray, idx: np.ndarray, update_step: int):


#         if update_step < self.num_batch:
#             self.last_step[idx] = update_step
#             self.loss[idx] = loss

#         elif update_step < self.num_batch*2:
#             self.grad[idx] = (loss - self.loss[idx]) / np.expand_dims((update_step - self.last_step[idx]), 1)
#             self.last_step[idx] = update_step
#             self.loss[idx] = loss
#         else:
#             dist = np.expand_dims((update_step - self.last_step[idx]), 1)
#             new_grad = (loss - self.loss[idx]) / dist
#             decay = self.decay ** dist
#             self.grad[idx] = self.grad[idx] * decay + new_grad * (1-decay)

   
#             weight = -self.grad[idx] * 50
#             weight = weight - np.max(weight)
#             weight = np.exp(weight)
#             norm_w  = weight / weight.sum()

#             idx = np.random.choice(weight.size, int(weight.size*0.75), False, np.reshape(norm_w, (-1,)))
#             # _, idx = torch.topk(torch.from_numpy(np.reshape(weight, (-1,))), int(weight.size*0.75))
#             # idx = idx.numpy()

            
#             mask = np.zeros(weight.size)
#             mask[idx] = 1
#             mask = np.reshape(mask, weight.shape)

#             return torch.from_numpy(mask).bool()

    
#     def sample(self):

#         idxes = np.random.choice(self.size, self.size, replace=False)

#         for i in range(self.num_batch):
#             batch_idxes = idxes[i*self.batch_size:(i+1)*self.batch_size]
#             yield torch.from_numpy(self.data[batch_idxes].astype(np.int64)), torch.from_numpy(batch_idxes)

#     def __len__(self):
#         return self.num_batch
    
#     def seq_sample(self):
#         num_batches = self.size // self.batch_size

#         for i in range(num_batches):
#             idxes = np.arange(i*self.batch_size,(i+1)*self.batch_size)
#             yield torch.from_numpy(self.data[idxes].astype(np.int64))

# class Buffer:
#     def __init__(self, data, batch_size) -> None:
        
#         seq_len = data.shape[1]-1

#         self.batch_size = batch_size
#         self.num_batch = data.shape[0] // batch_size
 
#         self.data = data[:self.batch_size*self.num_batch]
#         # self.data = data[:2048]

#         self.size = self.data.shape[0]

#         self.last_step = np.zeros(self.size, dtype=np.uint32)
#         self.loss = np.zeros(self.size, dtype=np.float16)
#         self.grad = np.zeros(self.size, dtype=np.float16)

#         self.decay = 0.9996

#     def update(self, loss: np.ndarray, idx: np.ndarray, update_step: int):


#         if update_step < self.num_batch:
#             self.last_step[idx] = update_step
#             self.loss[idx] = loss

#         elif update_step < self.num_batch*2:
#             # assert not ((idx - self.last_step[idx]) == 0).any()
#             self.grad[idx] = (loss - self.loss[idx]) / (update_step - self.last_step[idx])
#             self.last_step[idx] = update_step
#             self.loss[idx] = loss
#         else:
#             dist = (update_step - self.last_step[idx])
#             new_grad = (loss - self.loss[idx]) / dist
#             decay = self.decay ** dist
#             self.grad[idx] = self.grad[idx] * decay + new_grad * (1-decay)

#             # weight = -self.grad[idx] * 500
#             # weight = np.exp(weight)
#             # norm_w  = weight / weight.sum()
            

#             weight = -self.grad[idx] * 1000
#             weight = np.exp(weight)
#             norm_w  = weight / weight.sum()

#             if np.isnan(norm_w).any():
#                 print(-self.grad[idx] * 500)
#                 print(weight)
#                 print(norm_w)
#             # _, idx = torch.topk(torch.from_numpy(np.reshape(weight, (-1,))), int(weight.size*0.7))

#             idx = np.random.choice(16, 12, False, norm_w)
#             mask = np.zeros((16, 512))
#             mask[idx] = 1
#             # mask = np.reshape(mask, weight.shape)

#             return torch.from_numpy(mask).bool()

    
#     def sample(self):

#         idxes = np.random.choice(self.size, self.size, replace=False)

#         for i in range(self.num_batch):
#             batch_idxes = idxes[i*self.batch_size:(i+1)*self.batch_size]
#             yield torch.from_numpy(self.data[batch_idxes].astype(np.int64)), torch.from_numpy(batch_idxes)

#     def __len__(self):
#         return self.num_batch
    
#     def seq_sample(self):
#         num_batches = self.size // self.batch_size

#         for i in range(num_batches):
#             idxes = np.arange(i*self.batch_size,(i+1)*self.batch_size)
#             yield torch.from_numpy(self.data[idxes].astype(np.int64))

# ema

# class Buffer:
#     def __init__(self, data, batch_size) -> None:
        
#         seq_len = data.shape[1]-1

#         self.batch_size = batch_size
#         self.num_batch = data.shape[0] // batch_size 
 
#         self.data = data[:self.batch_size*self.num_batch]

#         self.size = self.data.shape[0]

#         self.last_step = np.zeros(self.size, dtype=np.uint32)
#         self.est = np.zeros((self.size, seq_len), dtype=np.float16)

#         self.decay = 0.9995

#     def update(self, loss: np.ndarray, idx: np.ndarray, update_step: int):


#         dist = np.expand_dims((update_step - self.last_step[idx]), 1)
#         decay = self.decay ** dist
#         self.est[idx] = self.est[idx] * decay + loss * (1-decay)
#         self.last_step[idx] = update_step

#         if update_step >= self.num_batch*2:
#             weight = self.est[idx]

            
#             norm_w = weight / weight.sum()

#             # idx = np.random.choice(weight.size, int(weight.size*0.75), False, np.reshape(norm_w, (-1,)))

#             idx = np.random.choice(weight.size, int(weight.size*0.75), False, p=np.reshape(norm_w, (-1,)))
            
#             mask = np.zeros(weight.size)
#             mask[idx] = 1
#             mask = np.reshape(mask, weight.shape)

#             return torch.from_numpy(mask).bool()

    
#     def sample(self):

#         idxes = np.random.choice(self.size, self.size, replace=False)

#         for i in range(self.num_batch):
#             batch_idxes = idxes[i*self.batch_size:(i+1)*self.batch_size]
#             yield torch.from_numpy(self.data[batch_idxes].astype(np.int64)), torch.from_numpy(batch_idxes)

#     def __len__(self):
#         return self.num_batch
    
#     def seq_sample(self, batch_size=None):
#         if batch_size is None:
#             batch_size = self.batch_size
#         num_batches = self.size // batch_size

#         for i in range(num_batches):
#             idxes = np.arange(i*batch_size,(i+1)*batch_size)
#             yield torch.from_numpy(self.data[idxes].astype(np.int64))


# class Buffer:
#     def __init__(self, data, batch_size) -> None:
        
#         seq_len = data.shape[1]-1

#         self.batch_size = batch_size
#         self.num_batch = data.shape[0] // batch_size 
 
#         self.data = data[:self.batch_size*self.num_batch]

#         self.size = self.data.shape[0]

#         # self.last_step = np.zeros(self.size, dtype=np.uint32)
#         self.init_loss = np.zeros((self.size, seq_len), dtype=np.float16)


#     def update(self, loss: np.ndarray, idx: np.ndarray, update_step: int):


#         if update_step < self.num_batch:
#             self.init_loss[idx] = loss

#         if update_step >= self.num_batch*2:
#             weight = loss

#             norm_w = np.reshape(weight / weight.sum(), (-1,))

#             # idx = np.random.choice(weight.size, int(weight.size*0.75), False, np.reshape(norm_w, (-1,)))

#             idx = np.random.choice(weight.size, int(weight.size*0.75), False, p=norm_w)
            
#             mask = np.zeros(weight.size)
#             mask[idx] = 1
#             mask = np.reshape(mask, weight.shape)

#             selected_prob = norm_w[idx]


#             return torch.from_numpy(mask).bool()

    
#     def sample(self):

#         idxes = np.random.choice(self.size, self.size, replace=False)

#         for i in range(self.num_batch):
#             batch_idxes = idxes[i*self.batch_size:(i+1)*self.batch_size]
#             yield torch.from_numpy(self.data[batch_idxes].astype(np.int64)), torch.from_numpy(batch_idxes)

#     def __len__(self):
#         return self.num_batch
    
#     def seq_sample(self, batch_size=None):
#         if batch_size is None:
#             batch_size = self.batch_size
#         num_batches = self.size // batch_size

#         for i in range(num_batches):
#             idxes = np.arange(i*batch_size,(i+1)*batch_size)
#             yield torch.from_numpy(self.data[idxes].astype(np.int64))

# pretrained loss
class Buffer:
    def __init__(self, data, batch_size) -> None:
        
        seq_len = data.shape[1]-1

        self.batch_size = batch_size
        self.num_batch = data.shape[0] // batch_size
 
        self.data = data[:self.batch_size*self.num_batch]
        # print(self.data.shape)
        self.size = self.data.shape[0]

        self.loss = np.load('./best_loss.npy')

        self.drop_idx = []
        # print(self.loss.shape)


    def update(self, loss: np.ndarray, idx: np.ndarray, update_step: int):
        pass
        # best_loss = (loss - self.loss[idx]).mean(1)



        # drop_idx = np.argwhere(best_loss *5 <= np.random.rand(*best_loss.shape))
        

        # if drop_idx.size > 0:
        #     # print(idx)
        #     # print(best_loss)
        #     # print(drop_idx)
        #     # print(idx[drop_idx])
        #     self.drop_idx.append(idx[drop_idx])

        # return torch.from_numpy(mask).bool()

    
    def sample(self):

        idxes = np.random.choice(self.size, self.size, replace=False)
        # print()
        if len(self.drop_idx) > 0:
            print(f'drop: {len(self.drop_idx)}')
            idxes = np.setdiff1d(idxes, np.concatenate(self.drop_idx))
            self.drop_idx.clear()
            print(f'idxes: {idxes.size}')
        print(f'batch: {math.ceil(idxes.size/self.batch_size)}')
        for i in range(math.ceil(idxes.size/self.batch_size)):
            batch_idxes = idxes[i*self.batch_size:(i+1)*self.batch_size]
            yield torch.from_numpy(self.data[batch_idxes].astype(np.int64)), torch.from_numpy(batch_idxes)

    def __len__(self):
        return self.num_batch
    
    def seq_sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        num_batches = self.size // batch_size

        for i in range(num_batches):
            idxes = np.arange(i*batch_size,(i+1)*batch_size)
            yield torch.from_numpy(self.data[idxes].astype(np.int64))
#%%
import numpy as np
# %%
a = np.array([1,2,3])
if a.size > 0:
    a = np.setdiff1d(np.array([1,2,3]), np.array([2, 3]))
a
# %%
