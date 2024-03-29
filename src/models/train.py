import numpy as np
import time
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn

from src.constants import MODELS_PATH
from src.data.loader_factory import load_data
from src.options import Options
from src.models.encoder import EncoderCNN
from src.models.logs import (
    AverageMeter,
    Logger,
    AttentionLogger,
    EmbeddingLogger,
    InferenceLogger,
)
from src.models.loss import DetangledJoinDomainLoss
from src.models.utils import save_checkpoint, load_model
from src.models.test import test


def adjust_learning_rate(optimizer, epoch):
    """
    Updates the learning rate given an schedule and a gamma parameter.
    """
    if epoch in args.schedule:
        args.learning_rate *= args.gamma
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.learning_rate


def train(args, data_loader, model, optimizer, criterion, epoch):
    """
    Train an epoch
    """
    batch_time = AverageMeter()
    losses_dom = AverageMeter()
    losses_spa = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    im_net, sk_net = model
    if args.cuda:
        im_net, sk_net = im_net.cuda(), sk_net.cuda()
    im_net.train()
    sk_net.train()
    torch.set_grad_enabled(True)

    end = time.time()
    for i, (sk, im, im_neg, _, _) in enumerate(data_loader):

        # Prepare input data
        if args.cuda:
            im, im_neg, sk = im.cuda(), im_neg.cuda(), sk.cuda()
        optimizer.zero_grad()

        # Model Output: Image and sketch encoding and projection to semantic space
        # Image Positive
        im_feat, _ = im_net(im)
        # Image Negative
        im_feat_neg, _ = im_net(im_neg)
        # Sketch
        sk_feat, _ = sk_net(sk)

        # Loss
        loss, loss_dom, loss_spa = criterion(im_feat, sk_feat, im_feat_neg, i)

        # Gradients and update
        loss.backward()
        optimizer.step()

        # Save values
        losses_dom.update(loss_dom.item(), args.batch_size)
        losses_spa.update(loss_spa.item(), args.batch_size)
        losses.update(loss.item(), args.batch_size)
        batch_time.update(time.time() - end, args.batch_size)
        end = time.time()

    print(
        "Epoch: [{0}] Average Loss {loss.avg:.4f} ({loss_dom.avg:.4f} + {loss_spa.avg:.4f}); \
           Avg Time x Batch {b_time.avg:.3f}".format(
            epoch,
            loss=losses,
            loss_dom=losses_dom,
            loss_spa=losses_spa,
            b_time=batch_time,
        )
    )

    return losses, losses_dom, losses_spa


def main():
    """
    Full training pipeline
    """
    print("Prepare data")
    transform = transforms.Compose([transforms.ToTensor()])
    (
        train_data,
        [valid_sk_data, valid_im_data],
        [test_sk_data, test_im_data],
        dict_class,
    ) = load_data(args, transform)

    dataloader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.prefetch,
        "pin_memory": args.cuda,
        "drop_last": True,
        "shuffle": True,
    }
    train_loader = DataLoader(train_data, **dataloader_args)

    dataloader_args["shuffle"] = False
    dataloader_args["batch_size"] = 3 * args.batch_size
    valid_sk_loader = DataLoader(valid_sk_data, **dataloader_args)
    valid_im_loader = DataLoader(valid_im_data, **dataloader_args)
    test_sk_loader = DataLoader(test_sk_data, **dataloader_args)
    test_im_loader = DataLoader(test_im_data, **dataloader_args)

    if args.attn:
        attention_logger = AttentionLogger(
            valid_sk_data, valid_im_data, logger, dict_class, args
        )
        inference_logger = InferenceLogger(
            valid_sk_data, valid_im_data, logger, dict_class, args
        )
        embedding_logger = EmbeddingLogger(
            valid_sk_data, valid_im_data, logger, dict_class, args
        )

    print("Create trainable model")
    im_net = EncoderCNN(
        out_size=args.emb_size, pretrained=args.nopretrain, attention=args.attn
    )
    sk_net = EncoderCNN(
        out_size=args.emb_size, pretrained=args.nopretrain, attention=args.attn
    )

    print("Loss, Optimizer & Evaluation")
    criterion = DetangledJoinDomainLoss(
        emb_size=args.emb_size,
        w_dom=args.w_domain,
        w_spa=args.w_triplet,
        lambd=args.grl_lambda,
    )
    criterion.train()
    optimizer = torch.optim.SGD(
        list(im_net.parameters())
        + list(sk_net.parameters())
        + list(criterion.parameters()),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.decay,
        nesterov=True,
    )

    if args.cuda and args.ngpu > 1:
        print("\t* Data Parallel")
        im_net = nn.DataParallel(im_net, device_ids=list(range(args.ngpu)))
        sk_net = nn.DataParallel(sk_net, device_ids=list(range(args.ngpu)))
        criterion = nn.DataParallel(criterion, device_ids=list(range(args.ngpu)))

    if args.cuda:
        im_net, sk_net = im_net.cuda(), sk_net.cuda()
        criterion = criterion.cuda()

    start_epoch = 0
    best_map = 0
    early_stop_counter = 0
    if args.load is not None:
        print("Loading model")
        im_net, sk_net, criterion, start_epoch, best_map = load_model(
            args.load, im_net, sk_net, criterion
        )

    print("***Train***")
    for epoch in range(start_epoch, args.epochs):
        # Update learning rate
        adjust_learning_rate(optimizer, epoch)
        print(f"Training epoch {epoch}.")
        loss_train, loss_dom, loss_spa = train(
            args,
            train_loader,
            [im_net, sk_net],
            optimizer,
            criterion,
            epoch,
        )
        print(f"Validation epoch {epoch}.")
        map_valid, map_valid_200, prec_valid_200 = test(
            args,
            valid_im_loader, valid_sk_loader,
            [im_net, sk_net],
            inference_logger,
        )

        # Logger
        im_net.eval()
        sk_net.eval()
        if not args.attn:
            pass
        else:
            with torch.set_grad_enabled(False):
                im_net.to(torch.device("cpu"))
                sk_net.to(torch.device("cpu"))
                attention_logger.plot_attention(im_net, sk_net)
                embedding_logger.plot_embeddings(im_net, sk_net)

        # Scalars
        logger.add_scalar("loss_train", loss_train.avg)
        logger.add_scalar("loss_domain", loss_dom.avg)
        logger.add_scalar("loss_triplet", loss_spa.avg)
        logger.add_scalar("map_valid", map_valid)
        logger.add_scalar("map_valid_200", map_valid_200)
        logger.add_scalar("prec_valid_200", prec_valid_200)
        logger.add_scalar("learning_rate", args.learning_rate)
        logger.step()

        # Early-Stop
        if map_valid_200 > best_map:
            best_map = map_valid_200
            best_epoch = epoch + 1
            early_stop_counter = 0
            if args.save is not None:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "im_state": im_net.state_dict(),
                        "sk_state": sk_net.state_dict(),
                        "criterion": criterion.state_dict(),
                        "best_map": best_map,
                    },
                    directory=args.save,
                    file_name="checkpoint",
                )
        else:
            if early_stop_counter == args.early_stop:
                break
            early_stop_counter += 1

    # Load Best model in case of save it
    if args.save is not None:
        print("Loading best  model")
        best_model_file = os.path.join(args.save, "checkpoint.pth")
        im_net, sk_net, criterion, best_epoch, best_map = load_model(
            best_model_file, im_net, sk_net, criterion
        )

    print("***Test***")
    map_test, map_200, prec_200 = test(
        args, test_im_loader, test_sk_loader, [im_net, sk_net]
    )
    print("Test mAP {mean_ap}%".format(mean_ap=map_test))
    print("Test mAP@200 {map_200}%".format(map_200=map_200))
    print("Test Precision@200 {prec_200}%".format(prec_200=prec_200))

    with open(os.path.join(args.save, "results.txt"), "w") as fp:
        print("Epoch: {best_epoch:.3f}".format(best_epoch=best_epoch), file=fp)
        print("Valid: {mean_ap:.3f}".format(mean_ap=best_map), file=fp)
        print("Test mAP: {mean_ap:.3f}".format(mean_ap=map_test), file=fp)
        print("Test mAP@200: {map_200:.3f}".format(map_200=map_200), file=fp)
        print("Test Precision@200: {prec_200:.3f}".format(prec_200=prec_200), file=fp)


if __name__ == "__main__":
    # Parse options
    args = Options().parse()
    print("Parameters:\t" + str(args))

    # Check cuda & Set random seed
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()
    print("Cuda:\t" + str(args.cuda))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print("Initialize logger")
    args.save = MODELS_PATH + args.name
    if os.path.exists(args.save):
        raise FileExistsError(f"{args.save} directory already exists.")

    # Create logger
    print("Log dir:\t" + args.save)
    logger = Logger(args.save, force=True)
    with open(os.path.join(args.save, "params.txt"), "w") as fp:
        for key, val in vars(args).items():
            fp.write("{} {}\n".format(key, val))

    main()
