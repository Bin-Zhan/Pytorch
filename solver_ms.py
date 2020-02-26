import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import datetime

from ignite.contrib.handlers.param_scheduler import create_lr_scheduler_with_warmup, CosineAnnealingScheduler
from ignite.engine import Events, Engine
from ignite.handlers import Timer, ModelCheckpoint
from ignite import engine
from ignite.metrics import Loss, Accuracy
from ignite.utils import convert_tensor

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from lossutils import Arcface
from datautils import ItemList
from resnets import resnext101_32x8d



# helper functions
def prepare_batch(batch, device=None):

    x, y = batch

    return (convert_tensor(x, device=device), convert_tensor(y, device=device))


def create_trainer(model, optimizer, loss_fn, device=None):

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        inputs, labels = prepare_batch(batch, device=device)
        preds = model(inputs, labels)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)


def create_evaluator(model, metrics, device=None):

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            inputs, labels = prepare_batch(batch, device=device)
            preds = model(inputs)
            return preds, labels

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


# config
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# save model
t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
snapshots = '/mnt/dataserver/zhanbin/frs/snapshots/{}'.format(t)
if not os.path.exists(snapshots):
    os.makedirs(snapshots)

init_lr = 1e-1
end_lr = 1e-5
batch = 512
epochs = 20
workernum = 4
num_classes = 78924

train_txt = 'ms_train.txt'
val_txt = 'ms_valid.txt'

trainset = ItemList(train_txt, transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]))
trainloader = DataLoader(trainset,
    batch_size=batch,
    shuffle=True,
    num_workers=workernum,
    drop_last=True,
    pin_memory=True
)

valset = ItemList(val_txt, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]))
valoader = DataLoader(valset,
    batch_size=batch,
    shuffle=True,
    num_workers=workernum,
    drop_last=True,
    pin_memory=True
)

# nets
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.features = resnext101_32x8d()
        self.fc = Arcface(512, num_classes)

    def forward(self, x, label=None):

        features = self.features(x)
        out = self.fc(features, label)

        return out

model = Net()
print(model)
model.to(device)

# multi-gpus
if torch.cuda.device_count():
    print('==================== Use {} GPUs ===================='.format(torch.cuda.device_count()))
    model = nn.DataParallel(model)

# loss function
loss_fn = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=5e-4)

# scheduler
scheduler = CosineAnnealingScheduler(optimizer, 'lr', init_lr, end_lr, 4*len(trainloader), cycle_mult=1.5, start_value_mult=0.1)
scheduler = create_lr_scheduler_with_warmup(scheduler, warmup_start_value=0., warmup_end_value=init_lr, warmup_duration=len(trainloader))

# create trainer
trainer = create_trainer(model, optimizer, loss_fn, device=device)
trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

# add timer for each iteration
timer = Timer(average=False)

# logging training loss
def log_loss(engine):
    i = engine.state.iteration
    e = engine.state.epoch

    if i % 100 == 0:
        print('[Iters {:0>7d}/{:0>2d}, {:.2f}s/100 iters, lr={:.4E}] loss={:.4f}'.format(i, e, timer.value(), optimizer.param_groups[0]['lr'], engine.state.output))
        timer.reset()
trainer.add_event_handler(Events.ITERATION_COMPLETED, log_loss)

# Evaluation
metrics = {
    'loss': Loss(loss_fn),
    'acc': Accuracy()
}

def score_fn(engine):
    acc = engine.state.metrics['acc']

    return acc

evaluator = create_evaluator(model, metrics, device=device)

def log_metrics(engine):

    metrics = evaluator.run(valoader).metrics
    print('[INFO] Compute metrics...')
    print(' Validation Results - Average Loss: {:.4f} | Accuracy: {:.4f}'.format(metrics['loss'], metrics['acc']))
    print('[INFO] Complete metrics...')
trainer.add_event_handler(Events.EPOCH_COMPLETED, log_metrics)

# save the model checkpoints
saver = ModelCheckpoint(snapshots, 'r101', n_saved=10, score_name='acc', score_function=score_fn)
evaluator.add_event_handler(Events.COMPLETED, saver, {'model': model.module})

# start training
print('[INFO] Start training...')
trainer.run(trainloader, epochs)
print('[INFO] Complete training...')