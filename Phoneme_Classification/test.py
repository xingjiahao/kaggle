import torch
from model import Classifier

concat_nframes = 3   # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.75   # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = 1213          # random seed
batch_size = 512        # batch size
num_epoch = 15         # the number of training epoch
learning_rate = 1e-4      # learning rate
model_path = './model.ckpt'  # the path where the checkpoint will be saved
lambd = 0.1     # dropout keep probability

# model parameters
# TODO: change the value of "hidden_layers" or "hidden_dim" for medium baseline
input_dim = 39 * concat_nframes  # the input dim of the model, you should not change the value
hidden_layers = 7          # the number of hidden layers
hidden_dim = 64           # the hidden dim


model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim, lambd=lambd)

# for name, param in model.named_parameters():
#     print(name, param)

# print('---------------')

# model.load_state_dict(model_path)
# model.load_state_dict(torch.load(model_path))

# 加载之前保存的参数（来自 hidden_layers=3 的模型）
pretrained_dict = torch.load(model_path)
model_dict = model.state_dict()

# for name, param in pretrained_dict.items():
#     print(name, param)
# print("------------------")
# for name, param in model_dict.items():
#     print(name, param)
print(pretrained_dict.keys())
print(model_dict.keys())
# 过滤和更新模型参数
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)


# for name, param in model.named_parameters():
#     print(name, param)

# print('model construct is :')
# print(model)