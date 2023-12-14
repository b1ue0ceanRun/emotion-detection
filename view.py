import re
import matplotlib.pyplot as plt

def parse_log_file(log_file):
    epochs = []
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []

    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'TRAIN' in line and 'acc=' in line and 'Ls=' in line:
                epoch_match = re.search(r'\[([\d]+)/', line)
                acc_match = re.search(r'acc=([.\d]+)', line)
                loss_match = re.search(r'Ls=([.\d]+)', line)

                if epoch_match and acc_match and loss_match:
                    epoch = int(epoch_match.group(1))
                    acc = float(acc_match.group(1))
                    loss = float(loss_match.group(1))
                    epochs.append(epoch)
                    train_acc.append(acc)
                    train_loss.append(loss)
            elif 'VALID' in line and 'acc=' in line and 'Ls=' in line:
                acc_match = re.search(r'acc=([.\d]+)', line)
                loss_match = re.search(r'Ls=([.\d]+)', line)

                if acc_match and loss_match:
                    acc = float(acc_match.group(1))
                    loss = float(loss_match.group(1))
                    val_acc.append(acc)
                    val_loss.append(loss)

    return epochs, train_acc, train_loss, val_acc, val_loss
def plot_metrics(epochs, train_metric, val_metric, metric_name):
    plt.figure(figsize=(10, 6))

    # 绘制训练数据
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_metric, label=f'Train {metric_name}')
    plt.title(f'Training {metric_name} over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()

    # 绘制验证数据
    plt.subplot(2, 1, 2)
    val_epochs = range(1, len(val_metric) + 1)
    plt.plot(val_epochs, val_metric, label=f'Validation {metric_name}', color='orange')
    plt.title(f'Validation {metric_name} over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()

    plt.tight_layout()
    plt.show()

log_file = './logs/epoch60_DACL'  # 替换为您的日志文件路径
epochs, train_acc, train_loss, val_acc, val_loss = parse_log_file(log_file)

# 可视化准确性
plot_metrics(epochs, train_acc, val_acc, 'Accuracy')

# 可视化损失
plot_metrics(epochs, train_loss, val_loss, 'Loss')
