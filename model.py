import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import utils

from config import args
device = torch.device(args.device)

class Network(nn.Module):

    def __init__(self, input_shape, output_size, criterion, path):
        super(Network, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self._criterion = criterion
        self.optim_path = path

        self.feature = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear((input_shape[1]) * (input_shape[2]) * 16, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def test_loss(self, ts):
        self.eval()
        with torch.no_grad():
            input, target = utils.load_single_test_data(self.optim_path, ts, self.input_shape)
            logits = self(input)
            loss = self._criterion(logits, target).item()
            onehot_target = np.zeros((target.size(0), self.output_size))
            onehot_target[np.arange(target.size(0)), np.array(target)] = 1
            sm = nn.Softmax(dim=1)
            probs = sm(logits)
            max_idx = torch.argmax(probs, 1)
            onehot_predict = np.zeros((target.size(0), self.output_size))
            onehot_predict[np.arange(target.size(0)), np.array(max_idx)] = 1
            predict_correctness = np.sum(onehot_target * np.array(onehot_predict), axis=1).tolist()
            predict_confidence = np.sum(onehot_target * np.array(probs), axis=1).tolist()
        return loss, predict_correctness, predict_confidence

    def online_update(self, path, ts_list, input_shape, criterion, optimizer, logger, ga_gen):
        if len(ts_list) == 0:
            return 1
        self.train()
        train_data = utils.load_disc_update_data(path, ts_list, input_shape)
        train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                  shuffle=True, pin_memory=True, num_workers=2)

        logger.info('****************************************************************')
        logger.info('****************************************************************')
        logger.info('****************************************************************')
        logger.info('time = %s, ga gen %d, online update discriminator', str(utils.get_unix_timestamp()), ga_gen)

        for epoch in range(args.update_epochs):
            logger.info('time = %s, epoch %d', str(utils.get_unix_timestamp()), epoch)
            print('time = {}, epoch {}'.format(str(utils.get_unix_timestamp()), epoch))
            self.train()
            train_loss, train_acc = train(train_queue, self, criterion, optimizer, logger)
            logger.info('time = %s, train_loss %f train_acc %f', str(utils.get_unix_timestamp()), train_loss, train_acc)
            print(
                'time = {}, train_loss {} train_acc {}'.format(str(utils.get_unix_timestamp()), train_loss, train_acc))


def train(train_queue, model, criterion, optimizer, logger):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)
        input = Variable(input.float(), requires_grad=False).to(device)
        target = Variable(target, requires_grad=False).to(device)
        logits = model(input)
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))

        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)

        if step % args.report_freq == 0:
            logger.info('time = %s, train %03d %e %f', str(utils.get_unix_timestamp()), step, objs.avg, top1.avg)
            print('time = {}, train {} {}'.format(str(utils.get_unix_timestamp()), step, objs.avg))

    return objs.avg, top1.avg

def test(test_queue, model, criterion, logger):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(test_queue):
            n = input.size(0)
            input = Variable(input.float()).to(device)
            target = Variable(target).to(device)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))

            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)

            if step % args.report_freq == 0:
                logger.info('time = %s, test %03d %e %f', str(utils.get_unix_timestamp()), step, objs.avg, top1.avg)
                print('time = {}, test {} {}'.format(str(utils.get_unix_timestamp()), step, objs.avg))

    return objs.avg, top1.avg