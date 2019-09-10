import os
import sys
import numpy as np
import torch
import argparse

import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import tensorboardX

sys.path.insert(0,'../../')
import utils
from pretrain_model import Network

parser = argparse.ArgumentParser('discriminator')

# General settings
parser.add_argument('--device', type=str, default='cpu', help='device used for training')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')

# Training settings
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--port', type=int, default=23333, help='distributed port')

parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')

parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')

parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')

parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

parser.add_argument('--bn_affine', action='store_true', default=False, help='update para in BatchNorm or not')

# Discriminator settings
parser.add_argument('--num_drones', type=int, default=9, help='num of drones used in the set of simulations')
parser.add_argument('--observ_window', type=int, default=5, help='length of the discriminator observation in seconds')
parser.add_argument('--downsampling', type=int, default=1, help='downsampling rate of the observation')
parser.add_argument('--multi_slice', action='store_true', default=False,
                    help='whether use multiple slices from a single simulation')
parser.add_argument('--save', type=str, default='../discrim_pretrain_logs', help='experiment name')

args = parser.parse_args()

device = torch.device(args.device)

def main():
    seed = args.seed

    np.random.seed(seed)

    cudnn.benchmark = True
    torch.manual_seed(seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(seed)
    timestamp = str(utils.get_unix_timestamp())
    path = os.path.join(args.save, timestamp)
    logger = utils.get_logger(args.save, timestamp, file_type='txt')
    tb_logger = tensorboardX.SummaryWriter('../runs/{}'.format(timestamp))

    logger.info("time = %s, args = %s", str(utils.get_unix_timestamp()), args)

    train_data, test_data, input_shape = utils.get_data(args.data, args.observ_window, args.downsampling, args.multi_slice)

    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                              shuffle=True, pin_memory=True, num_workers=2)
    test_queue = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                             shuffle=False, pin_memory=True, num_workers=2)

    model = Network(input_shape, args.num_drones)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    for epoch in range(args.epochs):
        lr = scheduler.get_lr()[0]
        logger.info('time = %s, epoch %d lr %e', str(utils.get_unix_timestamp()), epoch, lr)
        print('time = {}, epoch {} lr {}'.format(str(utils.get_unix_timestamp()), epoch, lr))
        model.train()
        train_loss, train_acc = train(train_queue, model, criterion, optimizer, logger)
        logger.info('time = %s, train_loss %f train_acc %f', str(utils.get_unix_timestamp()), train_loss, train_acc)
        print('time = {}, train_loss {} train_acc {}'.format(str(utils.get_unix_timestamp()), train_loss, train_acc))
        tb_logger.add_scalar("epoch_train_loss", train_loss, epoch)
        tb_logger.add_scalar("epoch_train_acc", train_acc, epoch)

        scheduler.step()

        model.eval()
        test_loss, test_acc = test(test_queue, model, criterion, logger)
        logger.info('time = %s, test_loss %f test_acc %f', str(utils.get_unix_timestamp()), test_loss, test_acc)
        print('time = {}, test_loss {} test_acc {}'.format(str(utils.get_unix_timestamp()), test_loss, test_acc))
        tb_logger.add_scalar("epoch_test_loss", test_loss, epoch)
        tb_logger.add_scalar("epoch_test_acc", test_acc, epoch)

        utils.save(model, os.path.join(path, 'weights.pt'))

def train(train_queue, model, criterion, optimizer, logger):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)
        input = Variable(input.float(), requires_grad=False).to(device)
        target = Variable(target.long(), requires_grad=False).to(device)
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

    with torch.no_grad():
        for step, (input, target) in enumerate(test_queue):
            n = input.size(0)
            input = Variable(input.float()).to(device)
            target = Variable(target.long()).to(device)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))

            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)

            if step % args.report_freq == 0:
                logger.info('time = %s, test %03d %e %f', str(utils.get_unix_timestamp()), step, objs.avg, top1.avg)
                print('time = {}, test {} {}'.format(str(utils.get_unix_timestamp()), step, objs.avg))

    return objs.avg, top1.avg

if __name__ == '__main__':
    main()

