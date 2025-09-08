import matplotlib.pyplot as plt
import pandas as pd

print("starting")

# Path to your CSV file
csv_file = "finetunings/gemini/Loss (2).csv"   

# Load CSV
df = pd.read_csv(csv_file)
# Show original column names (for debugging)
print("Original columns:", df.columns.tolist())

# Rename columns for convenience
df.columns = ["step", "train_loss", "val_loss"]

# Clean data
df.replace("undefined", pd.NA, inplace=True)
df["step"] = df["step"].astype(int)
df["train_loss"] = pd.to_numeric(df["train_loss"], errors="coerce")
df["val_loss"] = pd.to_numeric(df["val_loss"], errors="coerce")

# Plot
plt.figure(figsize=(12,6))
plt.plot(df["step"], df["train_loss"], label="Training Loss", marker="o")
plt.plot(df["step"], df["val_loss"], label="Validation Loss", linestyle="--", marker="x")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss (from CSV)")
plt.legend()
plt.grid(True)

# Save
output_file = "finetunings/gemini/train_vs_val_loss_ds.png"
plt.savefig(output_file, dpi=300)

plt.show()
