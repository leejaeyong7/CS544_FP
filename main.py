from os import path
import argparse
import logging

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import utils, CrossEntropyLoss
from torch import optim

from model import MiniModel as MiniNet
from model_saver import CheckPoint
from torchvision import datasets, transforms

logging.basicConfig(level='INFO',
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

transform = transforms.Compose([
    transforms.ToTensor()
])

loss_fun = CrossEntropyLoss()
iter_step = 0


def main(args):
    run_name = args.run_name
    gpus = args.gpu
    save_step = args.save_step
    is_train = args.train
    epochs = args.epochs
    checkpoint_dir = args.checkpoint_path
    grad_clip = args.gradient_clip
    resume = args.resume
    optimizer_type = args.optimizer
    batch_size = args.batch_size

    learning_rate = args.learning_rate
    scheduler_step = args.scheduler_step
    scheduler_gamma = args.scheduler_gamma
    scheduler_end = args.scheduler_end


    writer = SummaryWriter(comment='/runs/{}'.format(run_name))
    latest_checkpoint_name = '{}-latest.ckpt'.format(run_name)
    latest_checkpoint_path = path.join(checkpoint_dir, latest_checkpoint_name)

    ##################################
    # -- setup dataloader / variables
    if(gpus != None):
        device = torch.device('cuda:{}'.format(gpus))
    else:
        device = torch.device('cpu')

    # if last checkpoint exists, load from path
    if((not path.exists(latest_checkpoint_path)) or (not resume)):
        model = MiniNet().to(device)
        if(is_train):
            if(optimizer_type == 'sgd'):
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            else:
                optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, history_size=30)

            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=scheduler_step,
                                                  gamma=scheduler_gamma)
        total_step = 0
    else:
        checkpoint = CheckPoint.load(latest_checkpoint_path, device)
        model = MiniNet().to(device)
        model.load_state_dict(checkpoint['model'])
        if(is_train):
            optimizer = checkpoint['optimizer']
            scheduler = checkpoint['scheduler']
            epoch = checkpoint['epoch']
            total_step = checkpoint['total_step']
        else:
            total_step = 0

    #################################
    # -- setup datasets
    dataset = datasets.CIFAR10('./data',
                             train=is_train,
                             download=True,
                             transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True,
                             num_workers=1,
                             pin_memory=True)

    #####################
    # -- Actual training
    if(is_train):
        model.train()
    else:
        model.eval()
    for epoch in range(epochs):
        total_correct = 0
        total_total = 0
        num_batches = len(data_loader)
        for i, data in enumerate(data_loader):
            if(is_train):
                if(total_step < scheduler_end):
                    scheduler.step()

            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            classes = torch.zeros_like(output)
            expected_labels = output.max(1)[1]
            num_correct = (labels == expected_labels).sum()
            num_total = labels.shape[0]
            percent_correct = float(num_correct) / float(num_total)


            total_correct += num_correct
            total_total += num_total

            writer.add_scalar('output/percent_correct', percent_correct, total_step)
            if(is_train):
                if(optimizer_type == 'sgd'):
                    output = model(images)
                    loss = loss_fun(output, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    message = '[Training] Step: {:06d}, Loss: {:04f}'
                    logging.info(message.format(total_step, loss.item()))
                    # clip gradient if grad_clip is given
                    if(grad_clip):
                        utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                    # write to tensorboard
                    writer.add_scalar('data/loss', loss, total_step)
                else:
                    global iter_step
                    def closure():
                        global iter_step
                        iter_step += 1
                        output = model(images)
                        loss = loss_fun(output, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        message = '[Training] Total Step: {:06d}, Iter Step: {:06d} Loss: {:04f}'
                        # write to tensorboard
                        writer.add_scalar('data/loss', loss, iter_step)
                        logging.info(message.format(total_step, iter_step, loss.item()))
                        # clip gradient if grad_clip is given
                        if(grad_clip):
                            utils.clip_grad_norm_(model.parameters(),
                                                  grad_clip)

                        return loss

                    optimizer.step(closure)

                # -- save the run every some time
                if((total_step) % save_step == 0):
                    checkpoint_name = '{}-{}.ckpt'.format(run_name, total_step)
                    checkpoint_path = path.join(checkpoint_dir, checkpoint_name)
                    CheckPoint.save(checkpoint_path, model, optimizer, scheduler, total_step, epoch)
                    CheckPoint.save(latest_checkpoint_path, model, optimizer, scheduler, total_step, epoch)

                    # write historgram
                    for name, param in model.named_parameters():
                        writer.add_histogram(name, param.clone().cpu().data.numpy(), total_step)

            else:
                logging.info('[{:06d}/{:06d}] completed'.format(i+1, num_batches))
            # update optimizer
            total_step += 1
        writer.add_scalar('output/final_correct', float(total_correct) / float(total_total), epoch)
    writer.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', type=str, required=True, help='theme of this run')
    parser.add_argument('--optimizer', type=str, required=True, choices=['lbfgs', 'sgd'], help='type of optimizer')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID used for this run. Default=CPU')
    parser.add_argument('--checkpoint-path', type=str, default='./CheckPoint/', help='Path of checkpoint')
    parser.add_argument('--train', dest='train', action='store_true', help='train model')
    parser.add_argument('--resume', dest='resume', action='store_true', help='Resume from previous model')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Do not Resume from previous model')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--save-step', type=int, default=1000, help='Recurring number of steps for saving model')
    parser.add_argument('--epochs', type=int, default=300, help='Number of Epochs to run')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--gradient-clip', type=float, default=None, help='learning rate')
    parser.add_argument('--scheduler-step', type=int, default=2000, help='Scheduler step index value')
    parser.add_argument('--scheduler-end', type=int, default=10000, help='Scheduler final step value')
    parser.add_argument('--scheduler-gamma', type=float, default=0.2, help='Scheduler step update ratio')
    parser.set_defaults(resume=True, train=False)
    args = parser.parse_args()
    main(args)
