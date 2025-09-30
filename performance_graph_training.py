import matplotlib.pyplot as plt
from blackjack_training import history_v1
from blackjack_training_v2 import history_v2
from blackjack_training_v3 import history_v3

v1_train_acc = history_v1.history['accuracy'][-1]
v1_val_acc = history_v1.history['val_accuracy'][-1]
v2_train_acc = history_v2.history['accuracy'][-1]
v2_val_acc = history_v2.history['val_accuracy'][-1]
v3_train_acc = history_v3.history['accuracy'][-1]
v3_val_acc = history_v3.history['val_accuracy'][-1]

categories = ['Training Accuracy', 'Validation Accuracy']
v1_acc = [v1_train_acc, v1_val_acc]
v2_acc = [v2_train_acc, v2_val_acc]
v3_acc = [v3_train_acc, v3_val_acc]

bar_width = 0.25
x = range(len(categories))

plt.figure(figsize=(10, 6))
plt.bar(x, v1_acc, width=bar_width, label='v1', alpha=0.7)
plt.bar([p + bar_width for p in x], v2_acc, width=bar_width, label='v2', alpha=0.7)
plt.bar([p + 2 * bar_width for p in x], v3_acc, width=bar_width, label='v3', alpha=0.7)

plt.title('Model Accuracy Comparison')
plt.xlabel('Accuracy Type')
plt.ylabel('Accuracy')
plt.xticks([p + bar_width for p in x], categories)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()