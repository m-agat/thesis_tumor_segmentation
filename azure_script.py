from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment, Dataset
from azureml.core.compute import ComputeTarget

# Workspace from config.json file
ws = Workspace.from_config()

# Comupte Instance
target = "brats-training"
compute_target = ComputeTarget(workspace=ws, name=target)
print("Using existing compute target:", target)

# Load the environment
env = Environment.get(workspace=ws, name="brats-env")

# Getting the (registered) data by name
dataset = Dataset.get_by_name(workspace=ws, name='brats-dataset')
print("Dataset retrieved successfully!")

# Mounting the dataset
data_reference = dataset.as_mount()

# Configuring the script
config = ScriptRunConfig(
    source_directory='/home/agata/Desktop/thesis_tumor_segmentation/src',
    script='train/train.py',
    arguments=['--data_path', data_reference],
    compute_target=compute_target,
    environment=env,
)

# Creating  the experiment
exp = Experiment(ws, 'thesis_first_stage_train')

# Submitting the run
run = exp.submit(config)

# Printing out the details and waiting for completion
print(run)
run.wait_for_completion(show_output=True)
