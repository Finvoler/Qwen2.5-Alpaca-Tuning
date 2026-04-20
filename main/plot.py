import json
import numpy as np
import matplotlib.pyplot as plt

with open("training_loss.json", "r") as f:
    loss_history = json.load(f)

steps = [entry['step'] for entry in loss_history]
losses = [entry['loss'] for entry in loss_history]

plt.figure(figsize=(12, 6))

# 绘制原始 Loss (浅色)
plt.plot(steps, losses, label='Batch Loss', color='lightblue', alpha=0.6)

# 绘制移动平均 Loss (深色，更平滑)
if len(losses) > 10:
    window_size = 20 
    moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
    plt.plot(steps[window_size-1:], moving_avg, label=f'Moving Average (window={window_size})', color='red', linewidth=2)

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.savefig("loss_curve.png", dpi=300)
print("Loss curve plot saved to 'loss_curve.png'")