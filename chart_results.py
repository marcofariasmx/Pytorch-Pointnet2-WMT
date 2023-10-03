import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def inLine(line, string):
    """
    Tells if a part of a string is in another

    Parameters
    ----------
    line : str
        The full string
    string : str
        The part to find
    """
    return line.find(string) != -1


def setFloat(line, label, values):
    """
    If the line of the log file has a certain label nte following value is added to a list

    Parameters
    """
    if inLine(line, label):
        values.append(float(line.split(label)[1].split(' ')[0]))


def plot_graph(graph: Axes, label, ylabel, values, names):
    """
    Plots a subplot

    Parameters
    ----------
    graph : Axes
    label : str
        title of the graph
    ylabel : str
        Y axis label
    values : Dict[str: List[float]]
        Dictionary of values based on name
    names : List[str]
        Names of the different value groups
    """
    for name in names:
        graph.plot(values[name], marker='o', linestyle='-', label=name)
    graph.set_title(label)
    # graph.set_xlabel('Epoch')
    graph.set_ylabel(ylabel)
    graph.grid(True)
    plt.legend()


def plot_results(
    max_epoch,
    files,
    names,
    show_learning_rate: bool = False,
    show_train_accuracy: bool = False,
    show_mIOUs: bool = False,
    show_test_accuracy: bool = False,
    show_class_avg: bool = False,
    show_instance_avg: bool = False,
    show_best_accuracy: bool = False,
    show_best_class_avg: bool = False,
    show_best_instance_avg: bool = False
):
    """
    This method reads a log file and plots the values that are printed

    Parameters
    ----------
    max_epoch : int
        stoping point
    files : List[str]
        log files to read
    names : List[str]
        Names for the logs
    show_learning_rate : bool
    show_train_accuracy: bool
    show_mIOUs: bool
    show_test_accuracy: bool
    show_class_avg: bool
    show_instance_avg: bool
    show_best_accuracy: bool
    show_best_class_avg: bool
    show_best_instance_avg: bool
    """
    learning_rate = {name: [] for name in names}
    train_accuracy = {name: [] for name in names}
    mIOUs = {}
    test_accuracy = {name: [] for name in names}
    class_avg = {name: [] for name in names}
    instance_avg = {name: [] for name in names}
    best_accuracy = {name: [] for name in names}
    best_class_avg = {name: [] for name in names}
    best_instance_avg = {name: [] for name in names}

    lists = [
        learning_rate,
        train_accuracy,
        test_accuracy,
        class_avg,
        instance_avg,
        best_accuracy,
        best_class_avg,
        best_instance_avg,
    ]

    for idx, filename in enumerate(files):
        # Open the file for reading
        name = names[idx]
        with open(filename, 'r') as file:
            for line in file:
                show_learning_rate and setFloat(line, 'Learning rate:', learning_rate[name])

                show_train_accuracy and setFloat(line, 'Train accuracy is: ', train_accuracy[name])

                if show_mIOUs and inLine(line, 'eval mIoU of '):
                    line_items = line.split('eval mIoU of ')[1].split(' ')
                    label, rate = list(filter(lambda string: bool(string), line_items))
                    if not mIOUs.get(label):
                        mIOUs[label] = {name: [] for name in names}
                    mIOUs[label][name].append(float(rate))

                show_test_accuracy and setFloat(line, 'test Accuracy: ', test_accuracy[name])
                show_class_avg and setFloat(line, 'Class avg mIOU: ', class_avg[name])
                show_instance_avg and setFloat(line, 'Inctance avg mIOU: ', instance_avg[name])
                show_best_accuracy and setFloat(line, 'Best accuracy is: ', best_accuracy[name])
                show_best_class_avg and setFloat(line, 'Best class avg mIOU is: ', best_class_avg[name])
                show_best_instance_avg and setFloat(line, 'Best inctance avg mIOU is: ', best_instance_avg[name])

                if max_epoch:
                    check_lists = [l[name] for l in lists] + [v[name] for v in list(mIOUs.values())]
                    if any([len(l) > max_epoch for l in check_lists]):
                        break

    # Create a figure with subplots for both Train accuracy and eval mIoU
    i = 0
    shows = [
        show_learning_rate, show_train_accuracy, show_test_accuracy, show_class_avg,
        show_instance_avg, show_best_accuracy, show_best_class_avg, show_best_instance_avg
    ]
    plot_count = len(list(filter(lambda v: v, shows))) + len(mIOUs.keys())
    fig, axes = plt.subplots(plot_count, 1, figsize=(10, 6), sharex=True)

    # TRAIN ACCURACY
    if show_train_accuracy:
        plot_graph(axes[i], 'Train Accuracy Over Time', 'Accuracy', train_accuracy, names)
        i += 1

    # LEARNING RATE
    if show_learning_rate:
        plot_graph(axes[i], 'Learning Rate Over Time', 'Rate', learning_rate, names)
        i += 1

    # TEST ACCURACY
    if show_test_accuracy:
        plot_graph(axes[i], 'Test Accuracy Over Time', 'Accuracy', test_accuracy, names)
        i += 1

    # CLASS AVERAGE
    if show_class_avg:
        plot_graph(axes[i], 'Class Average mIOU', 'mIOU', class_avg, names)
        i += 1

    # INSTANCE AVERAGE
    if show_instance_avg:
        plot_graph(axes[i], 'Instance Average mIOU', 'mIOU', instance_avg, names)
        i += 1

    # BEST ACCURACY
    if show_best_accuracy:
        plot_graph(axes[i], 'Best Accuracy', 'Rate', best_accuracy, names)
        i += 1

    # BEST CLASS AVERAGE
    if show_best_class_avg:
        plot_graph(axes[i], 'Best Class Average mIOU', 'mIOU', best_class_avg, names)
        i += 1

    # BEST INSTANCE AVERAGE
    if show_best_instance_avg:
        plot_graph(axes[i], 'Best Instance Average mIOU', 'mIOU', best_instance_avg, names)
        i += 1

    # SHAPE mIOUs
    if show_mIOUs:
        for label, values in mIOUs.items():
            plot_graph(axes[i], label + ' mIOU', 'mIOU', values, names)
            i += 1

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    files = [
                'C:/Users/mmorgan/levvel/Pytorch-Pointnet2-WMT/log/part_seg/pointnet2_part_seg_msg/logs/pointnet2_part_seg_msg.txt',
                'C:/Users/mmorgan/levvel/Pytorch-Pointnet2-WMT/log/part_seg/pointnet2_part_seg_msg/logs/pointnet2_part_seg_msg1.txt',
                'C:/Users/mmorgan/levvel/Pytorch-Pointnet2-WMT/log/part_seg/pointnet2_part_seg_msg/logs/pointnet2_part_seg_msg2.txt',
                'C:/Users/mmorgan/levvel/Pytorch-Pointnet2-WMT/log/part_seg/pointnet2_part_seg_msg/logs/pointnet2_part_seg_msg3.txt'
            ]
    labels = ['Current Round', 'Test Facility As Is', 'Test Facility With Fix', 'Watson With Fix']
    plot_results(
        100,
        files,
        labels,
        show_mIOUs=True,
    )
    plot_results(
        100,
        files,
        labels,
        show_learning_rate=True,
        show_instance_avg=True,
        show_class_avg=True,
    )
    plot_results(
        100,
        files,
        labels,
        show_test_accuracy=True,
        show_train_accuracy=True,
    )
    plot_results(
        100,
        files,
        labels,
        show_best_accuracy=True,
        show_best_class_avg=True,
        show_best_instance_avg=True
    )
