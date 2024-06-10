import json
import matplotlib.pyplot as plt

file_path = './output/tulu_v2_gpt2/trainer_state.json'

# Load the JSON data from the file
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract losses and steps
losses = [entry.get('loss', None) for entry in data['log_history'] if 'loss' in entry]
steps = [entry.get('step', None) for entry in data['log_history'] if 'loss' in entry]

# Plot the losses
plt.figure(figsize=(10, 5))
plt.plot(steps, losses, marker='o', markersize=3, linestyle='-', color='blue')
plt.title('Loss vs. Steps')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.grid(True)

plt.savefig('loss_vs_steps.png', dpi=300, bbox_inches='tight')

plt.show()