# plot_loss.py

import pandas as pd
import matplotlib.pyplot as plt

def plot_loss(file="training_losses.xlsx"):
    df = pd.read_excel(file)
    plt.figure(figsize=(10, 6))
    plt.plot(df["train_loss_total"], label="Train")
    plt.plot(df["val_loss_total"], label="Validation")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss (Log Scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.show()

if __name__ == "__main__":
    plot_loss()