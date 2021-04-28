## Introduction

In Azure we can run experiments on script to pass data, train models and perfrom other data science and machine learning tasks. When running an experiment, we should consider the experiment run consists of two elements: the environment for the script which is depended on the language and the platform you use, and the compute target on which the environment will be deployed.

In this article, we will discuss how to create and use environment and compute targets in Azure machine learning platform.

## 1. Environment

Python code runs in the context of a virtual environment that defines the version of the Python runtime to be used as well as the installed packages available to the code. In most Python installations, packages are installed and managed in environments using Conda or pip.

To improve portability, we usually create environments in docker containers that are in turn be hosted in compute targets, such as your development computer, virtual machines, or clusters in the cloud.

See a structure of compute target:

![image](https://user-images.githubusercontent.com/71245576/116323460-040aaa80-a78c-11eb-8756-ad2528154235.png)

In general Azure machine learning handles environment creation and package installation through the creation of Docker containers. You can specify the Conda or pip packages you need and create an environment for the experiment.

In an enterprise machine learnig solution, experiments may be run in a variety of compute contexts, the environments are encapsulated by the Environment class in which you can use to create environments and specify runtime configuration for an experiment.

Now let's create an environment from a specification file, you can use a Conda or pip specification file to define the packages required in a Python environment and use it to create an Environment object. For example, save the following Conda configuratin settings in a file named conda.yml.

```Bash
name: py_env
dependencies:
  - numpy
  - pandas
  - scikit-learn
  - pip:
    - azureml-defaults
```
 And use the following code to create an Azure machine learning environment from the saved specification file:
 ```python
 from azureml.core import Environment

env = Environment.from_conda_specification(name='training_environment',
                                           file_path='./conda.yml')
```

My result:

![image](https://user-images.githubusercontent.com/71245576/116324156-95c6e780-a78d-11eb-920a-2e921e9e1d28.png)

If you have an existing Conda environment defined on your workstation you also can use it to define an Azure machine learning environment:

```python
from azureml.core import Environment

env = Environment.from_existing_conda_environment(name='training_environment',
                                                  conda_environment_name='py_env')
```

Then you can specify the Conda and pip packages you need in an environment by using a CondaDependencies object like this:
```python
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

env = Environment('training_environment')
deps = CondaDependencies.create(conda_packages=['scikit-learn','pandas','numpy'],
                                pip_packages=['azureml-defaults'])
env.python.conda_dependencies = deps
```

Usually, you should create environments in containers (this is the default unless the docker.enabled property is set to False, in which case the environment is created directly in the compute target).

So you can enable the docker.enabled property is set to False like this:
```python
env.docker.enabled = True
```

Alternatively, have an image creaated on-demand based on the base image and additional settings in a dockerfile:

```python
env.docker.base_image = None
env.docker.base_dockerfile = './Dockerfile'
```

If your image already includes an installation of Python with the dependencies you need, you can override this behavior by setting python.user_managed_dependencies to True and setting an explicit Python path for your installation

```python
env.python.user_managed_dependencies=True
env.python.interpreter_path = '/opt/miniconda/bin/python'
```

After you have created an environment, you can register it in your workspace and reuse it for future experiments that have the same Python dependencies. Now let's use the register method of an Environment object to register an environment:
```python
env.register(workspace=ws)
```

It has registered:

![image](https://user-images.githubusercontent.com/71245576/116325264-c1e36800-a78f-11eb-8907-8641241f8649.png)

Review the registered environments in your workspace like this:

```python
from azureml.core import Environment

env_names = Environment.list(workspace=ws)
for env_name in env_names:
    print('Name:',env_name)
```

There are parts of environments in my workspace:

![image](https://user-images.githubusercontent.com/71245576/116325353-f5be8d80-a78f-11eb-927e-fafd1e48ceae.png)

You can retrieve a registered environment by using the get method of the Environment class, and then assign it to a ScriptRunConfig.

For example, retrieve the training_environment registered environment and assign it to a script run configuration:

```python
from azureml.core import Environment, ScriptRunConfig

training_env = Environment.get(workspace=ws, name='training_environment')

script_config = ScriptRunConfig(source_directory='my_dir',
                                script='script.py',
                                environment=training_env)
```                                

When an experiment based on the estimator is run(you do not define an environment), Azure Machine Learning will look for an existing environment that matches the definition, and if none is found a new environment will be created based on the registered environment specification.

## 2. Compute targets

In tutorial, the definition of Compute Targets is the physical or virtual computers on which experiments are run. 

There are types of compute, you can select most appropriatet ype of compute target for the requirement.

![image](https://user-images.githubusercontent.com/71245576/116325763-f0ae0e00-a790-11eb-9b62-f8d5062a2621.png)

I think the code can be developed and tested on local or low-cost compute and them moved to more scalable compute for production workloads. You can run individual processes on the compute target that best fits its needs.

In Azure machine learning, you can take advantage of managing costs by paying only for what you use, it starts on-demand and stop automatically when no longer required, it also scales automatically based on workload processing needs.

Now let's create a compute target. The most common ways to create or attach a compute target are to use the Compute page in Azure machine learning studio or to use the Azure machine learning SKD to provivision compute targets in code. 

When using SDK, you can create a managed compute target with SDK. Specifically, to create an Azure machine learning compute cluster, you can use the azureml.core.compute.ComputeTarget class and the AmlCompute class:

```python
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute

# Load the workspace from the saved config file
ws = Workspace.from_config()

# Specify a name for the compute (unique within the workspace)
compute_name = 'aml-cluster'

# Define compute configuration
compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2',
                                                       min_nodes=0, max_nodes=4,
                                                       vm_priority='dedicated')

# Create the compute
aml_cluster = ComputeTarget.create(ws, compute_name, compute_config)
aml_cluster.wait_for_completion(show_output=True)
```
In this example, a cluster with up to four nodes that is based on the STANDARD_DS12_v2 virtual machine image will be created. The priority for the virtual machines (VMs) is set to dedicated, meaning they are reserved for use in this cluster (the alternative is to specify lowpriority, which has a lower cost but means that the VMs can be preempted if a higher-priority workload requires the compute).

You also can attach an unmanaged compute target with the SDK using the ComputeTarget.attch() method to attach the existing compute based on its target-specific configuration settings. An unmanaged compute target is one that is defined and managed outside of the Azure Machine Learning workspace; for example, an Azure virtual machine or an Azure Databricks cluster.

```python
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, DatabricksCompute

# Load the workspace from the saved config file
ws = Workspace.from_config()

# Specify a name for the compute (unique within the workspace)
compute_name = 'db_cluster'

# Define configuration for existing Azure Databricks cluster
db_workspace_name = 'db_workspace'
db_resource_group = 'db_resource_group'
db_access_token = '1234-abc-5678-defg-90...'
db_config = DatabricksCompute.attach_configuration(resource_group=db_resource_group,
                                                   workspace_name=db_workspace_name,
                                                   access_token=db_access_token)

# Create the compute
databricks_compute = ComputeTarget.attach(ws, compute_name, db_config)
databricks_compute.wait_for_completion(True)
```

You can check for an existing compute target or create a new one if there is not already one with the specified name:

```python
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

compute_name = "aml-cluster"

# Check if the compute target exists
try:
    aml_cluster = ComputeTarget(workspace=ws, name=compute_name)
    print('Found existing cluster.')
except ComputeTargetException:
    # If not, create it
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2',
                                                           max_nodes=4)
    aml_cluster = ComputeTarget.create(ws, compute_name, compute_config)

aml_cluster.wait_for_completion(show_output=True)
```

In your compute targets, you can use them to run specific workloads; such as experiments. To use a particular compute target, you can specify it in the appropriate parameter for an experiment run configuration or estimator. For example, the following code configures an estimator to use the compute target named aml-cluster:

```python
from azureml.core import Environment, ScriptRunConfig

compute_name = 'aml-cluster'

training_env = Environment.get(workspace=ws, name='training_environment')

script_config = ScriptRunConfig(source_directory='my_dir',
                                script='script.py',
                                environment=training_env,
                                compute_target=compute_name)
```

You can specify a ComputeTarget object instead of the compute target as well:

```python
from azureml.core import Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget

compute_name = "aml-cluster"

training_cluster = ComputeTarget(workspace=ws, name=compute_name)

training_env = Environment.get(workspace=ws, name='training_environment')

script_config = ScriptRunConfig(source_directory='my_dir',
                                script='script.py',
                                environment=training_env,
                                compute_target=training_cluster)
```

## Reference:

Build AI solutions with Azure Machine Learning, retrieve from https://docs.microsoft.com/en-us/learn/paths/build-ai-solutions-with-azure-ml-service/
