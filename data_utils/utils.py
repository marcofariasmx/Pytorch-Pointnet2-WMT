def split_list(input_list, split_percentage: float = .7):
    """
    Method to create training and testing datasets lists based on lists of facilities provided

    If a -1 is given as input for split_percentage, then it proceeds to return the input list repeated.
    """
    if split_percentage == -1 or len(input_list) == 1:
        return input_list, input_list

    if split_percentage < 0 or split_percentage > 1:
        raise ValueError("Split percentage should be between 0 and 1.")

    split_index = int(len(input_list) * split_percentage)

    # Must have at least 1 item in each list
    if split_index == len(input_list):
        split_index = split_index - 1

    return input_list[:split_index], input_list[split_index:]