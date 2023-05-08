from model import *

# data prarameters
# TODO: change the value of "concat_nframes" for medium baseline
concat_nframes = 11   # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.75   # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = 1213          # random seed
batch_size = 512        # batch size
num_epoch = 5         # the number of training epoch
learning_rate = 1e-3      # learning rate
model_path = './model1.ckpt'  # the path where the checkpoint will be saved
lambd = 0.1     # dropout keep probability

# model parameters
# TODO: change the value of "hidden_layers" or "hidden_dim" for medium baseline
input_dim = 39 * concat_nframes  # the input dim of the model, you should not change the value
hidden_layers = 16          # the number of hidden layers
hidden_dim = 64           # the hidden dim


same_seeds(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')

def getDataset(concat_nframes,train_ratio,seed):
    # preprocess data
    train_X, train_y = preprocess_data(split='train', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio, random_seed=seed)
    val_X, val_y = preprocess_data(split='val', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio, random_seed=seed)

    # get dataset
    train_set = LibriDataset(train_X, train_y)
    val_set = LibriDataset(val_X, val_y)

    # remove raw feature to save memory
    del train_X, train_y, val_X, val_y
    gc.collect()
    # get dataloader
    return train_set,val_set

train_set,val_set=getDataset(concat_nframes,train_ratio,seed)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# create model, define a loss function, and optimizer
# model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim, lambd=lambd).to(device)

def get_net(input_dim, hidden_layers, hidden_dim, lambd, device, model_path):
    model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim, lambd=lambd).to(device)
    if os.path.exists(model_path):
        # 加载模型
        # model.load_state_dict(torch.load(model_path))
        pretrained_dict = torch.load(model_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("模型加载成功")
    else:
        print("模型不存在")
    return model

model=get_net(input_dim, hidden_layers, hidden_dim, lambd, device, model_path)

def train(model, train_loader, val_loader, train_set, val_set, num_epoch, learning_rate, model_path):
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_acc = 0.0
    for epoch in range(num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        
        # training
        model.train() # set the model to training mode
        for i, batch in enumerate(tqdm(train_loader)):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad() 
            outputs = model(features) 
            
            loss = criterion(outputs, labels)
            loss.backward() 
            optimizer.step() 
            
            _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
            train_acc += (train_pred.detach() == labels.detach()).sum().item()
            train_loss += loss.item()
        
        # validation
        model.eval() # set the model to evaluation mode
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader)):
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                
                loss = criterion(outputs, labels) 
                
                _, val_pred = torch.max(outputs, 1) 
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                val_loss += loss.item()

        print(f'[{epoch+1:03d}/{num_epoch:03d}] Train Acc: {train_acc/len(train_set):3.5f} Loss: {train_loss/len(train_loader):3.5f} | Val Acc: {val_acc/len(val_set):3.5f} loss: {val_loss/len(val_loader):3.5f}')

        # if the model improves, save a checkpoint at this epoch
        # if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), model_path)
        print(f'saving model with acc {best_acc/len(val_set):.5f}')

    del train_set, val_set
    del train_loader, val_loader
    gc.collect()

train(model, train_loader, val_loader, train_set, val_set, num_epoch, learning_rate,model_path)

# load data
# test_X = preprocess_data(split='test', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes)
# test_set = LibriDataset(test_X, None)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# # load model
# model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
# model.load_state_dict(torch.load(model_path))

# pred = np.array([], dtype=np.int32)

# model.eval()
# with torch.no_grad():
#     for i, batch in enumerate(tqdm(test_loader)):
#         features = batch
#         features = features.to(device)

#         outputs = model(features)

#         _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
#         pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

# with open('prediction.csv', 'w') as f:
#     f.write('Id,Class\n')
#     for i, y in enumerate(pred):
#         f.write('{},{}\n'.format(i, y))