from sketchy_extended import Sketchy_Extended


def load_data(args, transform=None):
    """
    Load the data of the appropriate dataset
    Args:
        - args: metadata provided in src/options.py
        - transform: pytorch transform to process the data
    Return:
        -
        -
        -
        -
        -
        -
    """
    if args.dataset == "sketchy_extend":
        return Sketchy_Extended(args, transform)
    elif args.dataset == "tuberlin_extend":
        # TODO PML-09.02.2021
        return Sketchy_Extended(args, transform)
        # return TUBerlin_Extended(args, transform)
    else:
        sys.exit()

    raise NameError(args.dataset + " not implemented!")


if __name__ == "__main__":
    from src.options import Options
    import matplotlib.pyplot as plt
    from torchvision import transforms

    # Parse options
    args = Options().parse()
    print("Parameters:\t" + str(args))

    transform = transforms.Compose([transforms.ToTensor()])
    (
        train_loader,
        [valid_sk_loader, valid_im_loader],
        [test_sk_loader, test_im_loader],
        dict_class,
    ) = load_data(args, transform)

    print("\n--- Train Data ---")
    print("\t* Length: {}".format(len(train_loader)))
    print("\t* Classes: {}".format(train_loader.get_class_dict()))
    print("\t* Num Classes. {}".format(len(train_loader.get_class_dict())))

    print("\n--- Valid Data ---")
    print("\t* Length Sketch: {}".format(len(valid_sk_loader)))
    print("\t* Length Image: {}".format(len(valid_im_loader)))
    print("\t* Classes: {}".format(valid_sk_loader.get_class_dict()))
    print("\t* Num Classes. {}".format(len(valid_sk_loader.get_class_dict())))

    print("\n--- Test Data ---")
    print("\t* Length Sketch: {}".format(len(test_sk_loader)))
    print("\t* Length Image: {}".format(len(test_im_loader)))
    print("\t* Classes: {}".format(test_sk_loader.get_class_dict()))
    print("\t* Num Classes. {}".format(len(test_sk_loader.get_class_dict())))
