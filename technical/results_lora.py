import json
import os
import matplotlib.pyplot as plt

import os
import json

# Find the latest checkpoint folder
checkpoint_folders = [f for f in os.listdir("results_lora") if f.startswith("checkpoint-")]
checkpoint_folders.sort(key=lambda x: int(x.split("-")[1]))  # sort by step
latest_ckpt = checkpoint_folders[-1]

trainer_file = os.path.join("results_lora", latest_ckpt, "trainer_state.json")

with open(trainer_file, "r") as f:
    trainer_state = json.load(f)

# Now you can extract log_history and plot as before
losses = trainer_state.get("log_history", [])

for entry in losses[-10:]:
    print(entry)

# Extract loss values
loss_values = [e["loss"] for e in losses if "loss" in e]

# Plot
plt.figure(figsize=(10,6))
plt.plot(loss_values, label="Training Loss")
plt.xlabel("Logging Step")
plt.ylabel("Loss")
plt.title("LoRA Fine-tuning Loss")
plt.legend()
plt.grid(True)

# Save plot to PNG
output_file = "results_lora/lora_training_loss.png"
plt.savefig(output_file, dpi=300)  # dpi=300 for high quality
print(f"✅ Loss plot saved to {output_file}")

# Show the plot
plt.show()
