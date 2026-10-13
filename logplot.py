# plot_logs.py
import json, sys, matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for path in sys.argv[1:]:
    log = json.load(open(path))
    name = path.split("/")[-1].replace(".json", "")
    axes[0].plot(log["iter"], log["train_loss"], label=name, alpha=0.7)
    axes[1].plot(log["val_iter"], log["val_loss"], label=name, marker="o")

axes[0].set(xlabel="iter", ylabel="train loss", title="Train loss")
axes[1].set(xlabel="iter", ylabel="val loss",   title="Val loss")
axes[0].set_yscale("log"); axes[1].set_yscale("log")
axes[0].legend(); axes[1].legend()
plt.tight_layout(); plt.savefig("loss_curves.png", dpi=150)
print("saved loss_curves.png")