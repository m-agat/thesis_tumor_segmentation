from azureml.core import (
    Workspace,
    Experiment,
    ScriptRunConfig,
    Environment,
    Dataset,
    Datastore,
)

ws = Workspace.from_config()

# Create a new environment from the modified YAML file
new_env = Environment.from_conda_specification(
    name="brats-env",
    file_path="/home/agata/Desktop/thesis_tumor_segmentation/conda.yml",
)

# Register the new environment version
new_env.register(workspace=ws)

# ws = Workspace.from_config()

# # Get the existing environment by name
# env = Environment.get(workspace=ws, name='my-env')  # Replace with your environment name

# # Save the environment to a directory, which will include the conda.yml file
# env.save_to_directory(path='./brats-env-yml')
