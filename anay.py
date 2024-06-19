import numpy as np

steps = 20
seq_len = 512

step_idx = []
loss = []


for i in range(steps):
    idx = np.load(f'./loss/{i+1}_idx.npy')
    l = np.load(f'./loss/{i+1}_loss.npy')
    step_idx.append(np.stack([idx for _ in range(seq_len)], -1))
    loss.append(l)

    step_idx.append(idx)
    loss.append(l)




    # if i == 2:


step_idx = np.stack(step_idx, -1)
print(step_idx.shape)
step_idx = np.reshape(step_idx, (1024*seq_len, steps))
loss = np.stack(loss, -1)
loss = np.reshape(loss, (1024*seq_len, steps))
grad = np.stack(grad, -1)
print(grad.shape)
grad = np.reshape(grad, (1024*seq_len, steps-1))



num_samples = 10
sample_idx = np.random.choice(step_idx.shape[0], num_samples, False)

# print(grad.shape)
# # %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))

# Plot all rows in the same figure
for i in range(num_samples):
    ax.plot(step_idx[sample_idx[i]], loss[sample_idx[i]], label=f'Plot {i+1}')

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.5)

# Display the plot
plt.show()
# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))

# Plot all rows in the same figure
for i in range(num_samples):
    # print(grad[i])
    val = np.exp(-grad[sample_idx[i]] * 100)
    ax.scatter(range(steps-1), val, label=f'Plot {i+1}')

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.5)

# Display the plot
plt.show()
# %%
import numpy as np
# %%
last_loss = np.load('./loss/19_loss.npy').mean(1)
for i in range(18):
    l = np.load(f'./loss/{i+1}_loss.npy').mean(1)
    print(((l - last_loss )< 0.1).mean())

# %%
