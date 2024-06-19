import matplotlib.pyplot as plt
import json
with open('./baseline.json') as f:
    baseline = json.load(f)
baseline = [b[2] for b in baseline]

with open('./best_loss_select.json') as f:
    rho = json.load(f)
rho = [b[2] for b in rho]

plt.plot(list(range(len(baseline))), baseline, list(range(len(rho))), rho)
plt.show()