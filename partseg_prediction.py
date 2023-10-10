"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.ShapeNetDataLoader import PartNormalDataset
from data_utils.utils import split_list
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
import platform
import numpy as np
import laspy
import pandas as pd

# Get the system's platform
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


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--log_dir', type=str, default='pointnet2_part_seg_msg', help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    parser.add_argument('--num_workers', type=int, default=0, help='number of cpu threads to process data')
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_msg', help='model name')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'],
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--npoint', type=int, default=500,
                        help='number of points to process for each rack per chunk/batch')
    parser.add_argument('--points_per_scan', type=int, default=1000000,
                        help='number of points to load for each scan\'s pointcloud')
    parser.add_argument('--train_test_split', type=float, default=0.7,
                        help='Fraction of facilities to use for training vs. testing')
    parser.add_argument('--facilities_dirs', type=str, nargs='+', default=None,
                        help='1 to N directories containing facilities, e.g. c:/users/me/Data/Facility1 c:/users/me/Data/Facility2 d:/data/Facility1')
    parser.add_argument('--data_dir', type=str, help='Directory containing several facilities')

    return parser.parse_args()


def pop_from_array(arr, n_items):
    """
    Pops the first n_items from the array.

    Parameters:
    - arr: The input ndarray.
    - n_items: The number of items to pop.

    Returns:
    - popped_items: The first n_items from the array.
    - arr: The remaining items after popping.
    """
    popped_items = arr[:n_items]
    arr = arr[n_items:]
    return popped_items, arr


def save_labeled_las(facility: Facility, labeled_points_indices, labeled_data):
    """
    Save the labeled point cloud of a facility as a .las file.

    Parameters:
    - facility (Facility): An instance containing the scan data for the facility.
    - labeled_points_indices: Indices indicating which points in the facility's point cloud are labeled.
    - labeled_data: 3D data structure containing labeling information for the specified indices.

    Procedure:
    1. Extracts the facility's point cloud data and initializes the 'point_label' column with 0s.
    2. Transposes labeled_data for ease of labeling extraction.
    3. Extracts and adjusts labels from the transposed labeled_data.
    4. Assigns these labels to the facility's point cloud data at the specified indices.
    5. Initializes a .las file header with an additional dimension for 'point_label'.
    6. Populates the new .las file with the updated point cloud data.
    7. Determines the filename for saving based on the facility's attributes.
    8. Checks for the existence of a similarly named file and notifies the user.
    9. Saves the .las file.
    """

    # Extract the facility's point cloud and initialize 'point_label' column to 0 (everything)
    facility_point_cloud_df = facility.scans[0].point_cloud.points
    facility_point_cloud_df['point_label'] = 0

    # Transpose labeled_data for extraction
    labeled_data = labeled_data.transpose(0, 2, 1)

    # Extract and adjust labels from the transposed data
    labeled_values = []
    for i in range(labeled_data.shape[0]):
        for j in range(labeled_data.shape[1]):
            points = labeled_data[i, j]
            labeled_values.append(points[-1])

    # Add a +1 to shift values. 0 + 1 = clutter, 1+1 = shelf, 2+1 = poles, etc.
    labeled_values = [x + 1 for x in labeled_values]

    # Assign labels to the facility's point cloud data
    facility_point_cloud_df.loc[labeled_points_indices, 'point_label'] = labeled_values

    # 1. Initialize a .las file header with an extra dimension for 'point_label'
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="point_label", type=np.int32))

    # 2. Create a new .las file and populate it with the updated data
    las = laspy.LasData(header)

    las.x = (facility_point_cloud_df['x']).astype(np.int32)
    las.y = (facility_point_cloud_df['y']).astype(np.int32)
    las.z = (facility_point_cloud_df['z']).astype(np.int32)
    las.classification = facility_point_cloud_df['class'].astype(np.uint8)
    las.red = facility_point_cloud_df['r'].astype(np.uint16)
    las.green = facility_point_cloud_df['g'].astype(np.uint16)
    las.blue = facility_point_cloud_df['b'].astype(np.uint16)
    las.user_data = facility_point_cloud_df['user_data'].astype(np.uint8)
    las.intensity = facility_point_cloud_df['intensity'].astype(np.uint16)
    las.point_label = facility_point_cloud_df['point_label'].astype(np.uint8)

    # 3. Determine the filename based on the facility's attributes
    facility_filename = os.path.join(facility.data_path, facility.default_name+'-labeled_points.las')

    # Check and notify if the file already exists
    if os.path.exists(facility_filename):
        print(f"'{facility_filename}' exists!")
    else:
        print(f"'{facility_filename}' does not exist!")

    # Save the .las file
    las.write(facility_filename)


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    experiment_dir = 'log/part_seg/' + args.log_dir

    '''HYPER PARAMETER'''
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError(
            "CUDA is not available. Please use --device cpu or make sure CUDA is installed and compatible.")

    if args.device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
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

        # Check for JSONs
        if len(facilities_jsons) == 0:
            print('Could not find any merged JSON files')
            exit(3)

    # Split the facilities into training and testing data
    _, test_set_facilities = split_list(facilities_jsons, split_percentage=args.train_test_split)

    print("Test set facilities: \n", test_set_facilities)

    # Load the data
    test_facilities = []

    for file in tqdm(test_set_facilities, desc="Loading Test Facilities", unit="facility"):
        facility = Facility(files=file, points_per_scan=args.points_per_scan)
        test_facilities.append(facility)

    TEST_DATASET = RackPartSegDataset(facilities=test_facilities, points_per_chunk=args.npoint, include_bulk=False)

    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers)

    # create dictionary to associate facilities to their racks to later produce the labeled las file
    facility_shapes_dict = TEST_DATASET.facility_shapes_dict

    log_string("The number of test data is: %d" % len(TEST_DATASET))

    original_classes_dict = TEST_DATASET.get_classes()
    num_different_parts = len(TEST_DATASET.get_parts_dict())

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

    print(seg_label_to_cat)

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    if args.model == 'pointnet2_part_seg_msg':
        classifier = MODEL.get_model(num_classes=num_classes, num_parts=num_part, custom_data=custom_data,
                                     normal_channel=args.normal).to(device)
    else:
        classifier = MODEL.get_model(num_part, normal_channel=args.normal).to(device)
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

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

        pred_results = []
        racks_points_indices = []
        """
        Processing loop description:
        
        batch_id: each batch processes up to X amount of objects (racks) simultaneously
        
        points: points to be analized through the neuralnetwork
        label: part type (drive_in_rack, select_rack, etc).
        target: part segment type (clutter, shelf, pole, etc).
        points_indices: the indices of the processed points from the originally loaded point cloud
        """
        for batch_id, (points, label, target, points_indices) in tqdm(enumerate(testDataLoader),
                                                                      total=len(testDataLoader),
                                                                      smoothing=0.9):
            if not args.normal:  # if normals are not taken into account, only process the first 3 numbers (x,y,z).
                points = points[:, :, :3]
            batchsize, num_point, _ = points.size()
            cur_batch_size, NUM_POINT, _ = points.size()
            # points are actual points (xyz, rgb), label stands for the class num and target stands for the part num
            points, label, target = points.float().to(device), label.long().to(device), target.long().to(device)
            points = points.transpose(2, 1)
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).to(device)

            # Perform the prediction X amount of times (number of votes) in order to
            for _ in range(args.num_votes):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred

            # Get the average of the predictions.
            seg_pred = vote_pool / args.num_votes
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()

            # Convert the tensor to ndarray
            converted_points = points.numpy()

            # Expand dimensions of ndarray_b
            ndarray_b_exp = np.expand_dims(cur_pred_val, 1)

            # Concatenate along the second dimension
            points_pred_result = np.concatenate((converted_points, ndarray_b_exp), axis=1)

            # Append the result to the results list
            pred_results.append(points_pred_result)

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                # NOTE: In this following line is where the prediction happens.
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

            racks_points_indices.append(pd.Index(points_indices.numpy().tolist()))

        # After the loop, concatenate all the results along the first dimension #(N amount of racks, M amount of dims
        # (x,y,z,r,g,b,point_label), K amount of points per rack)
        final_pred_result = np.concatenate(pred_results, axis=0)

        # Combine the list of Index objects into a single Index object
        # 1. Flatten the nested list of indices from multiple racks into a single list
        combined_index = pd.Index([item for index in racks_points_indices for item in index])

        # 2. Convert a multi-level index to a flat index for consistency
        combined_index = combined_index.to_flat_index()

        # 3. Convert inner lists to tuples to maintain multi-dimensional indices structure
        combined_index = [tuple(item) if isinstance(item, list) else item for item in combined_index]

        # Process each facility with its newly labeled points and produce a new labeled las file
        for facility, racks_list in facility_shapes_dict.items():

            facility_labeled_points_indices, combined_index = pop_from_array(combined_index, len(racks_list))
            facility_labeled_data, final_pred_result = pop_from_array(final_pred_result, len(racks_list))

            flat_combined_index = [item for sublist in facility_labeled_points_indices for item in sublist]

            save_labeled_las(facility=facility, labeled_points_indices=flat_combined_index,
                             labeled_data=facility_labeled_data)

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
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

    log_string('Accuracy is: %.5f' % test_metrics['accuracy'])
    log_string('Class avg accuracy is: %.5f' % test_metrics['class_avg_accuracy'])
    log_string('Class avg mIOU is: %.5f' % test_metrics['class_avg_iou'])
    log_string('Inctance avg mIOU is: %.5f' % test_metrics['inctance_avg_iou'])


if __name__ == '__main__':
    args = parse_args()
    main(args)
