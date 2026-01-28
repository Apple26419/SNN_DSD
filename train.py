import datetime
from torch.utils.tensorboard import SummaryWriter
from utils.config import *
from utils.tvc import *
import collections
import torchattacks

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    args = get_args()
    train_data_loader, test_data_loader, c_in, num_classes = get_data(args.b, args.j, args.T, args.data_dir, args.dataset)
    net = get_net(args.surrogate, args.dataset, args.model, num_classes, args.drop_rate, args.tau, c_in)
    if args.attack == 'fgsm':
        attacker = torchattacks.FGSM(net, eps=args.eps / 255)
    elif args.attack == 'pgd':
        attacker = torchattacks.PGD(net, eps=args.eps / 255, alpha=args.alpha, steps=args.steps)
    else:
        attacker = None

    # optimizer preparing
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(args.opt)

    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'CosALR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
    else:
        raise NotImplementedError(args.lr_scheduler)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    # loading models from checkpoint
    start_epoch = 0
    max_val_acc = 0
    if args.resume:
        print('resuming...')
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_val_acc = checkpoint['max_val_acc']
        print('start epoch:', start_epoch, ', max validate acc:', max_val_acc)
    if args.pre_train:
        checkpoint = torch.load(args.pre_train, map_location='cpu')
        state_dict2 = collections.OrderedDict([(k, v) for k, v in checkpoint['net'].items()])
        net.load_state_dict(state_dict2)
        print('use pre-trained model, max validate acc:', checkpoint['max_val_acc'])

    out_dir = os.path.join(args.out_dir, f'{args.dataset}_{args.model}_T{args.T}_tau{args.tau}_e{args.epochs}_bs{args.b}_wd{args.weight_decay}_drop{args.drop_rate}')

    if args.attack is not None:
        if args.attack == 'fgsm':
            out_dir += f'_fgsm_eps{args.eps}'
        elif args.attack == 'pgd':
            out_dir += f'_pgd_eps{args.eps}_alpha{args.alpha}_steps{args.steps}'
        else:
            out_dir += f'_clean'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Make Dir {out_dir}.')
    else:
        print('out Dir already exists:', out_dir)

    # save the initialization of parameters
    if args.save_init:
        checkpoint = {
            'net': net.state_dict(),
            'epoch': 0,
            'max_val_acc': 0.0
        }
        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_0.pth'))
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
    writer = SummaryWriter(os.path.join(out_dir, 'logs'), purge_step=start_epoch)

    # training and validating
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = tra(model=net, dataset=args.dataset, data=train_data_loader, time_step=args.T, epoch=epoch, optimizer=optimizer, lr_scheduler=lr_scheduler, scaler=scaler, loss_lambda=args.loss_lambda, attacker=attacker, writer=writer)
        val_loss, val_acc = val(model=net, dataset=args.dataset, data=test_data_loader, time_step=args.T, epoch=epoch, optimizer=optimizer, lr_scheduler=lr_scheduler, scaler=scaler, loss_lambda=args.loss_lambda, attacker=attacker, writer=writer)

        save_max = False
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            save_max = True
        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_val_acc': max_val_acc
        }
        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        total_time = time.time() - start_time
        print(f'epoch={epoch}, train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, max_val_acc={max_val_acc:.4f}, total_time={total_time:.4f}, escape_time={(datetime.datetime.now() + datetime.timedelta(seconds=total_time * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}')
        if epoch == 0:
            print("Memory Reserved: %.4fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

if __name__ == '__main__':
    main()
