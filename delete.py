import os
import glob

# 目标目录列表
dirs = ['./logs/SGD_loss', './logs/SGD_acc', './logs/SGD_precision','./logs/SGD_recall',"./logs/SGD_f1",
        './logs/SGD-low_loss','./logs/SGD-low_acc','./logs/SGD-low_precision','./logs/SGD-low_recall','./logs/SGD-low_f1',
        './logs/SGD-momentum_loss','./logs/SGD-momentum_acc','./logs/SGD-momentum_precision','./logs/SGD-momentum_recall','./logs/SGD-momentum_f1',
        './logs/SGD-momentum-low_loss','./logs/SGD-momentum-low_acc','./logs/SGD-momentum-low_precision','./logs/SGD-momentum-low_recall','./logs/SGD-momentum-low_f1',
        './logs/Adam_loss','./logs/Adam_acc','./logs/Adam_precision','./logs/Adam_recall',"./logs/Adam_f1",
        './logs/RMSprop_loss','./logs/RMSprop_acc','./logs/RMSprop_precision','./logs/RMSprop_recall',"./logs/RMSprop_f1",
        './logs/Adagrad_loss','./logs/Adagrad_acc','./logs/Adagrad_precision','./logs/Adagrad_recall',"./logs/Adagrad_f1",
        './logs/Adagrad-low_loss', './logs/Adagrad-low_acc', './logs/Adagrad-low_precision', './logs/Adagrad-low_recall', './logs/Adagrad-low_f1',
        './logs/model']

# 遍历每个目录
for dir_path in dirs:
    # 获取目录中的所有文件
    files = glob.glob(os.path.join(dir_path, '*'))  # 匹配该目录下的所有文件

    # 删除每个文件
    for file in files:
        if os.path.isfile(file):  # 确保删除的是文件而非子目录
            os.remove(file)
            print(f"Deleted file: {file}")
