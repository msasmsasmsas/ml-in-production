# E:\ml-in-production\HW5_Training_Experiments\PR1\utils.py

def create_class_mapping(class_names):
    """
    Creates a mapping from class names to indices
    """
    return {cls: i for i, cls in enumerate(class_names)}