from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment, Dataset, Datastore 
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

# Get the model path
datastore = Datastore.get(ws, "workspaceblobstore")
model = Dataset.File.from_files(path=(datastore, 'directory/results/SwinUNetr'))
model_reference = model.as_mount()
print("Model retrieved successfully!")

segresnet = Dataset.File.from_files(path=(datastore, 'directory/results/SegResNet'))
segresnet_reference = segresnet.as_mount()

attunet = Dataset.File.from_files(path=(datastore, 'directory/results/AttentionUNet'))
attunet_reference = attunet.as_mount()

vnet = Dataset.File.from_files(path=(datastore, 'directory/results/VNet'))
vnet_reference = vnet.as_mount()

# Configuring the script
config = ScriptRunConfig(
    source_directory='/home/agata/Desktop/thesis_tumor_segmentation/src',
    script='ensemble/ensemble_with_meta_learner.py',
    arguments=['--data_path', data_reference,
               '--model_path', model_reference,
               '--swinunetr_path', model_reference,
               '--segresnet_path', segresnet_reference,
               '--attunet_path', attunet_reference,
               '--vnet_path', vnet_reference],
    compute_target=compute_target,
    environment=env,
)

# Creating  the experiment
exp = Experiment(ws, 'train_meta_learner')

# Submitting the run
run = exp.submit(config)

# Printing out the details and waiting for completion
print(run)
run.wait_for_completion(show_output=True)
