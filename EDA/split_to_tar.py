import os
import tarfile
import math


def split_folder_to_tar_gz(folder_path, output_prefix, num_parts=4):
    """
    Splits the contents of a folder into multiple .tar.gz files.

    Args:
        folder_path (str): The path to the folder to be split.
        output_prefix (str): The prefix for the output .tar.gz files.
        num_parts (int): Number of .tar.gz files to create.
    """
    # Get all files and directories in the folder
    items = os.listdir(folder_path)

    # Calculate chunk size
    chunk_size = math.ceil(len(items) / num_parts)

    # Split items into chunks
    chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    # Create tar.gz files for each chunk
    for i, chunk in enumerate(chunks):
        tar_filename = f"{output_prefix}_part{i + 1}.tar.gz"
        with tarfile.open(tar_filename, "w:gz") as tar:
            for item in chunk:
                item_path = os.path.join(folder_path, item)
                tar.add(item_path, arcname=item)  # Add to tar.gz with relative paths
                print(f"Added {item_path}")
        print(f"Created: {tar_filename}")


# Example usage
folder_to_split = "/home/magata/data/brats2021challenge/RelabeledTrainingData"  # Replace with your folder path
output_file_prefix = (
    "/home/magata/data/brats2021challenge/split_folder"  # Prefix for .tar.gz files
)
split_folder_to_tar_gz(folder_to_split, output_file_prefix, num_parts=4)
