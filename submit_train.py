from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment, Dataset
import azureml._restclient.snapshots_client

azureml._restclient.snapshots_client.SNAPSHOT_MAX_SIZE_BYTES = 2000000000

# Step 1: Load the workspace from config.json located in the conf/ directory
ws = Workspace.from_config(path="./conf/config.json")

# Step 2: Define the experiment
experiment_name = "my-segresnet-experiment"
experiment = Experiment(workspace=ws, name=experiment_name)

# Step 3: Define the compute target (your existing cluster)
compute_target = ws.compute_targets["brats2024-training-T4"]

# Step 4: Access the datastore and mount training and validation datasets
datastore = ws.datastores['workspaceblobstore']  # Change this to your appropriate datastore

# Create dataset references for training and validation data
train_dataset = Dataset.File.from_files(datastore.path('BraTS2024-BraTS-GLI-TrainingData/training_data1_v2'))
val_dataset = Dataset.File.from_files(datastore.path('BraTS2024-BraTS-GLI-ValidationData/validation_data'))

# Step 5: Create an environment from the Conda YAML file
env = Environment.from_conda_specification("my_custom_env", "./azureml-environment.yml")

# Step 6: Set up the ScriptRunConfig to run your train.py script
src = ScriptRunConfig(
    source_directory="./",
    script="train.py",
    compute_target=compute_target,
    environment=env,
    arguments=[
        '--config', './conf/config.yaml',  # Add config file
        '--train_data', train_dataset.as_mount(),  # Mount train data
        '--val_data', val_dataset.as_mount()  # Mount validation data
    ]
)

# Step 7: Submit the job
run = experiment.submit(src)
run.wait_for_completion(show_output=True)
