"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np

from pathlib import Path
from tqdm import tqdm
from data_utils.ShapeNetDataLoader import PartNormalDataset
from data_utils.utils import split_list

# Get the system's platform
import platform

system_platform = platform.system()

# Check if it's Windows or Linux
if system_platform == "Windows":
    print("The operating system is Windows.")
    sys_path = ['C:\\']


elif system_platform == "Linux":
    print("The operating system is Linux.")
    sys_path = ['/mnt', 'c']

else:
    print("The operating system is neither Windows nor Linux.")


"""
LabelPC Path:

*****Important for proper functioning of the program*****

Please make sure you modify this following line and change it to your current PointBluePython directory in your system
"""
labelpc_path = os.path.abspath('../PointBluePython')

if os.path.exists(labelpc_path):
    print(f"{labelpc_path} exists!")
else:
    print(f"Error: {labelpc_path} does not exist!")
    exit(1)

sys.path.append(labelpc_path)

try:
    from MachineLearningAutomation.Datasets import RackPartSegDataset
    from WarehouseDataStructures.Facility import Facility
    print("LabelPC Imports successful!")
except ImportError as e:
    print(f"Error during import: {e}")
    exit(2)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes, device):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.to(device)
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_msg', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='number of points to process per chunk/batch')
    parser.add_argument('--points_per_scan', type=int, default=1000000, help='number of points to load for each scan\'s pointcloud')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--num_workers', type=int, default=0, help='number of cpu threads to process data')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'],
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--train_test_split', type=float, default=0.7,
                        help='Fraction of facilities to use for training vs. testing')
    parser.add_argument('--facilities_dirs', type=str, nargs='+', default=None,
                        help='1 to N directories containing facilities, e.g. c:/users/me/Data/Facility1 c:/users/me/Data/Facility2 d:/data/Facility1')
    parser.add_argument('--data_dir', type=str, help='Directory containing several facilities')

    return parser.parse_args()

def get_json_files_path(facilities_path: str):
    facilities_list = os.listdir(facilities_path)

    json_files_path = []
    for annotated_facility in facilities_list:
        files_names = os.listdir(os.path.join(facilities_path, annotated_facility))
        for file_name in files_names:
            if file_name.endswith('.json'):
                json_files_path.append(os.path.join(facilities_path, annotated_facility, file_name))

    return json_files_path


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Please use --device cpu or make sure CUDA is installed and compatible.")

    if args.device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # Grab path to merged JSON file for each facility
    facilities_jsons = []

    if args.facilities_dirs:
        for facility_dir in args.facilities_dirs:
            merged_json = Facility.find_merged_json_file(facility_dir)
            if merged_json:
                facilities_jsons.append(merged_json)
            else:
                print(f'Could not find merged json file in {facility_dir}')

    elif args.data_dir:
        for facility_dir in os.listdir(args.data_dir):
            full_facility_path = os.path.join(args.data_dir, facility_dir)
            merged_json = Facility.find_merged_json_file(full_facility_path)
            if merged_json:
                facilities_jsons.append(merged_json)
            else:
                print(f'Could not find merged json file in {facility_dir}')

    else:
        print("No facilities directories given, starting training on test data")
        facility_dir = os.path.join(
            labelpc_path,
            'Test Facilities',
            'Test Facility - Annotated',
        )
        merged_json = Facility.find_merged_json_file(facility_dir)
        if merged_json:
            facilities_jsons.append(merged_json)
        else:
            print(f'Could not find merged json file in {facility_dir}')


        # # Custom testing data:
        # facilities1_path = sys_path + ['Users', 'M0x1', 'OneDrive', 'MachineLearningAutomation', 'FacilitiesX10New']
        # facilities1_path = os.path.join(*facilities1_path)
        # facilities1_list = get_json_files_path(facilities1_path)
        # facilities2_path = sys_path + ['Users', 'M0x1', 'Downloads', 'Facilities_NET_x31']
        # facilities2_path = os.path.join(*facilities2_path)
        # facilities2_list = get_json_files_path(facilities2_path)
        # train_set_facilities, test_set_facilities = split_lists([facilities1_list, facilities2_list], .7)


    # Check for JSONs
    if len(facilities_jsons) == 0:
        print('Could not find any merged JSON files')
        exit(3)

    # Split the facilities into training and testing data
    train_set_facilities, test_set_facilities = split_list(facilities_jsons, split_percentage=args.train_test_split)

    print("Train set facilities: \n", train_set_facilities)
    print("Test set facilities: \n", test_set_facilities)

    # Load the data
    train_facilities = []
    test_facilities = []

    for file in tqdm(train_set_facilities, desc="Loading Train Facilities", unit="facility"):
        facility = Facility(files=file, points_per_scan=args.points_per_scan)
        train_facilities.append(facility)

    for file in tqdm(test_set_facilities, desc="Loading Test Facilities", unit="facility"):
        facility = Facility(files=file, points_per_scan=args.points_per_scan)
        test_facilities.append(facility)

    TRAIN_DATASET = RackPartSegDataset(facilities=train_facilities, points_per_chunk=args.npoint, include_bulk=False)

    TEST_DATASET = RackPartSegDataset(facilities=test_facilities, points_per_chunk=args.npoint, include_bulk=False)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    original_classes_dict = TRAIN_DATASET.get_classes()
    num_different_parts = len(TRAIN_DATASET.get_parts_dict())

    """
    These numbers represent the original neural network design capabilities. It is able to handle 16 different
    classes with a total of 50 different parts across the classes. If they are to be modified, the NN structure
    has to change as well. For the moment being the train_partseg script allows to work with less classes and
    part numbers.
    """
    custom_data = True
    if custom_data:
        num_classes = len(original_classes_dict)
        num_part = num_different_parts * num_classes
    else:
        num_classes = 16
        num_part = 50

    seg_classes = {}
    counter = 0
    for key, value in original_classes_dict.items():
        seg_classes[key] = list(range(counter, counter + num_different_parts))
        counter += num_different_parts

    print(seg_classes)

    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    if args.model == 'pointnet2_part_seg_msg':
        classifier = MODEL.get_model(num_classes=num_classes, num_parts=num_part, custom_data=custom_data, normal_channel=args.normal).to(device)
    else:
        classifier = MODEL.get_model(num_part, normal_channel=args.normal).to(device)

    criterion = MODEL.get_loss().to(device)
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_instance_avg_iou = 0

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''learning one epoch'''
        for i, (points, label, target, points_indices) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            if not args.normal: #if normals are not taken into account, only process the first 3 numbers (x,y,z).
                points = points[:, :, :3]
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().to(device), label.long().to(device), target.long().to(device)
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes, device))
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))
            loss = criterion(seg_pred, target, trans_feat)
            loss.backward()
            torch.nn.utils.clip_grad_value_(classifier.parameters(), clip_value=1.0)
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        log_string('Train accuracy is: %.5f' % train_instance_acc)

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            classifier = classifier.eval()

            for batch_id, (points, label, target, points_indices) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().to(device), label.long().to(device), target.long().to(device)
                if not args.normal:  # if normals are not taken into account, only process the first 3 numbers (x,y,z).
                    points = points[:, :, :3]
                points = points.transpose(2, 1)
                seg_pred, _ = classifier(points, to_categorical(label, num_classes, device))
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()

                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=float))
            for cat in sorted(shape_ious.keys()):
                space_separator = 14
                log_string('eval mIoU of %s %f' % (cat + ' ' * (space_separator - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['instance_avg_iou'] = np.mean(all_shape_ious)

        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Instance avg mIOU: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['instance_avg_iou']))
        if (test_metrics['instance_avg_iou'] >= best_instance_avg_iou):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'instance_avg_iou': test_metrics['instance_avg_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['instance_avg_iou'] > best_instance_avg_iou:
            best_instance_avg_iou = test_metrics['instance_avg_iou']
        log_string('Best accuracy is: %.5f' % best_acc)
        log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        log_string('Best instance avg mIOU is: %.5f' % best_instance_avg_iou)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
