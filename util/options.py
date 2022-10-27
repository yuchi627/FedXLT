import argparse
import imp
import os
from Dataset.param_aug import ParamDiffAug
import pickle

def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--thres_many', type=int, default=100, help='The number of samples exceeds thres_many as Many-shot')
    parser.add_argument('--thres_few', type=int, default=20, help='The number of samples is less than thres_few as Few-shot')
    parser.add_argument('--path_place', type=str, default=os.path.join(path_dir, '../data/place/'))
    parser.add_argument('--path_cifar10', type=str, default=os.path.join(path_dir, '../data/CIFAR10/'))
    parser.add_argument('--path_cifar100', type=str, default=os.path.join(path_dir, '../data/CIFAR100/'))
    parser.add_argument('--path_covid', type=str, default=os.path.join(path_dir, '../data/COVID-19_Radiography_Dataset/'))
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_clients', type=int, default=20, help='number of client')
    parser.add_argument('--num_online_clients', type=int, default=8, help='Number of clients selected per round')
    parser.add_argument('--num_rounds', type=int, default=200)
    parser.add_argument('--batch_size_local_training', type=int, default=64)
    parser.add_argument('--batch_size_global_training', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=500)
    parser.add_argument('--local_epoch', type=int, default=10)  #
    parser.add_argument('--global_epoch', type=int, default=300)
    parser.add_argument('--freeze_epoch', type=int, default = 0, help='feature extractor start training after freeze_round. Only train classifier before freeze_epoch.')

    parser.add_argument('--fed_method', type=str, default='gkt', choices=['gkt', 'avg'],
                        help='use [avg] or [gkt] in federated learning ')
    parser.add_argument('--non_iid_alpha', type=float, default=0.5, help='Control the degree of heterogeneity')
    parser.add_argument('--seed', type=int, default=7, help='random seed')
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=0.02, type=float, help='Control the degree of imbalance.')
    parser.add_argument('--max_sample_proportion', type=int, default=1, help='The proportion of max number in oversampling [ max(num of sample in each class)/max_sample_proportion=resample number of every class ]')
    parser.add_argument('--save_ckpt', action='store_true', help='When set, save the checkpoint')  
    parser.add_argument('--detailed_eval', action='store_true', help='When set, write detailed eval info')  
    parser.add_argument('--long_tail', action='store_true', help='When set, the centralize training use long-tailed data')
    parser.add_argument('--iid', action='store_true', help='When set, use iid distribution, default non-iid')

    parser.add_argument('--named', type=str, default='prototree-fl', help='name of this model ')
    parser.add_argument('--comment', type=str, default='', help='some comment for this model, will show after model name')                        
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--workers', type=int, default=16, help='speed up dataloader')
    parser.add_argument('--check_size', action='store_true', help='When set, test')   
    parser.add_argument('--server_resampler',
                        action='store_true',
                        help='When set, the server resample data '
                        )
    parser.add_argument('--client_resampler',
                        action='store_true',
                        help='When set, the Client resample data'
                        )                        
    parser.add_argument('--cls_ser2cli',
                        action='store_true',
                        help='When set, the server trained classifier will replace client classifier to continue training'
                        )
    parser.add_argument('--classifier', type=str, default='tree', choices=['tree', 'linear'], help='use [tree] or [linear] to be the classifier ')
    parser.add_argument('--server_classifier', type=str, default='tree', choices=['tree', 'linear'])
    parser.add_argument('--client_classifier', type=str, default='tree', choices=['tree', 'linear'])
    parser.add_argument('--lamda_cls_num',
                        type=float,
                        default=0.0,
                        help='Ratio of fedavg weight [number of class]')
    parser.add_argument('--lamda_data_num',
                        type=float,
                        default=0.0,
                        help='Ratio of fedavg weight [number of data]')
    parser.add_argument('--lamda_data_rare',
                        type=float,
                        default=0.0,
                        help='Ratio of fedavg weight [rare of data]')


    parser.add_argument('--test_data', type=str, default='', help='Specifies the data distribution to use(if wanted)')
    parser.add_argument('--ckpt_dir', type=str, default='name of the checkpoint dir for loading')


    # prototree
    parser.add_argument('--net',
                        type=str,
                        default='resnet8',
                        help='Base network used in the tree. Pretrained network on iNaturalist is only available for resnet50_inat (default). Others are pretrained on ImageNet. Options are: resnet18, resnet34, resnet50, resnet50_inat, resnet101, resnet152, densenet121, densenet169, densenet201, densenet161, vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn or vgg19_bn')
    parser.add_argument('--num_features',
                        type=int,
                        default = 256,
                        help='Depth of the prototype and therefore also depth of convolutional output')
    parser.add_argument('--optimizer',
                        type=str,
                        default='AdamW',
                        help='The optimizer that should be used when training the tree')
    parser.add_argument('--milestones',
                        type=str,
                        default='60,70,80,90,100',
                        help='The milestones for the MultiStepLR learning rate scheduler')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.5,
                        help='The gamma for the MultiStepLR learning rate scheduler. Needs to be 0<=gamma<=1')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001, 
                        help='The optimizer learning rate for training the prototypes')
    parser.add_argument('--lr_block',
                        type=float,
                        default=0.001, 
                        help='The optimizer learning rate for training the 1x1 conv layer and last conv layer of the underlying neural network (applicable to resnet50 and densenet121)')
    parser.add_argument('--lr_net',
                        type=float,
                        default=0.01, 
                        # default=1e-5, 
                        help='The optimizer learning rate for the underlying neural network')
    parser.add_argument('--lr_pi',
                        type=float,
                        default=0.001, 
                        help='The optimizer learning rate for the leaf distributions (only used if disable_derivative_free_leaf_optim flag is set')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='The optimizer momentum parameter (only applicable to SGD)')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0,
                        help='Weight decay used in the optimizer')
    parser.add_argument('--disable_derivative_free_leaf_optim',
                        action='store_true',
                        help='Flag that optimizes the leafs with gradient descent when set instead of using the derivative-free algorithm'
                        )
    parser.add_argument('--kontschieder_train',
                        action='store_true',
                        help='Flag that first trains the leaves for one epoch, and then trains the rest of ProtoTree (instead of interleaving leaf and other updates). Computationally more expensive.'
                        )
    parser.add_argument('--kontschieder_normalization',
                        action='store_true',
                        help='Flag that disables softmax but uses a normalization factor to convert the leaf parameters to a probabilitiy distribution, as done by Kontschieder et al. (2015). Will iterate over the data 10 times to update the leaves. Computationally more expensive.'
                        )
    parser.add_argument('--log_probabilities',
                        action='store_true',
                        help='Flag that uses log probabilities when set. Useful when getting NaN values.'
                        )
    parser.add_argument('--pruning_threshold_leaves',
                        type=float,
                        default=0.01,
                        help='An internal node will be pruned when the maximum class probability in the distributions of all leaves below this node are lower than this threshold.')
    parser.add_argument('--nr_trees_ensemble',
                        type=int,
                        default=5,
                        help='Number of ProtoTrees to train and (optionally) use in an ensemble. Used in main_ensemble.py') 
    parser.add_argument('--img_size',
                        type=int,
                        default=224,
                        help='Number of worker to train and (optionally) use in an ensemble. Used in main_ensemble.py') 
    parser.add_argument('--disable_pretrained',
                        action='store_true',
                        help='When set, the backbone network is initialized with random weights instead of being pretrained on another dataset). When not set, resnet50_inat is initalized with weights from iNaturalist2017. Other networks are initialized with weights from ImageNet'
                        )
    parser.add_argument('--depth',
                        type=int,
                        default=4,
                        help='The tree is initialized as a complete tree of this depth')
    parser.add_argument('--W1',
                        type=int,
                        default = 1,
                        help='Width of the prototype. Correct behaviour of the model with W1 != 1 is not guaranteed')
    parser.add_argument('--H1',
                        type=int,
                        default = 1,
                        help='Height of the prototype. Correct behaviour of the model with H1 != 1 is not guaranteed')
    parser.add_argument('--dir_for_saving_images',
                        type=str,
                        default='upsampling_results',
                        help='Directoy for saving the prototypes, patches and heatmaps')
    parser.add_argument('--upsample_threshold',
                        type=float,
                        default=0.98,
                        help='Threshold (between 0 and 1) for visualizing the nearest patch of an image after upsampling. The higher this threshold, the larger the patches.')




    args = parser.parse_args()
    args.milestones = get_milestones(args)
    if args.net == "resnet8":
        args.ker_size=8
    else:
        args.ker_size=7


    return args

def get_milestones(args: argparse.Namespace):
    if args.milestones != '':
        milestones_list = args.milestones.split(',')
        for m in range(len(milestones_list)):
            milestones_list[m]=int(milestones_list[m])
    else:
        milestones_list = []
    return milestones_list

def save_args(args: argparse.Namespace, directory_path: str) -> None:
    """
    Save the arguments in the specified directory as
        - a text file called 'args.txt'
        - a pickle file called 'args.pickle'
    :param args: The arguments to be saved
    :param directory_path: The path to the directory where the arguments should be saved
    """
    # If the specified directory does not exists, create it
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    # Save the args in a text file
    with open(directory_path + '/args.txt', 'w') as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, str):  # Add quotation marks to indicate that the argument is of string type
                val = f"'{val}'"
            f.write('{}: {}\n'.format(arg, val))
            # print(arg, ": ", val)
    # Pickle the args for possible reuse
    # with open(directory_path + '/args.pickle', 'wb') as f:
    #     pickle.dump(args, f) 