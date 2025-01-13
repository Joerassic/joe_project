import os

from invoke import Context, task

"""
This script consists of task functions.
>invoke taskname
will perform the task
"""

# Initialisation
WINDOWS = os.name == "nt"
PROJECT_NAME = "joe_project"
PYTHON_VERSION = "3.11"


# ----- Setup tasks ----- #


# Creating a new environment
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


# Installing all the project requirements from requirements.txt
@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


# Installing developer requirements
@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)


# ----- Project tasks ----- #


# Process the data in the data/raw data folder and place inside data/processed folder
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"python src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)


# Train the CNN model on the training data (randomly split)
@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


# Evaluate the CNN model on the test data
@task
def evaluate(ctx: Context) -> None:
    """Evaluating model."""
    ctx.run(f"python src/{PROJECT_NAME}/evaluate.py models/model.pth", echo=True, pty=not WINDOWS)


# Create a figure of the accuracy and loss of the evaluation
@task
def visualize(ctx: Context) -> None:
    """Visualizing the model."""
    ctx.run(f"python src/{PROJECT_NAME}/visualize.py models/model.pth", echo=True, pty=not WINDOWS)


# Run tests
@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)


# Build docker
@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# ----- Documentation tasks ----- #


# Build documentation
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


# Serve documentation
@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)



# ----- Git tasks ----- #

# Add, commit and push with a message
@task
def git(ctx, message):
    ctx.run(f"git add .")
    ctx.run(f"git commit -m '{message}'")
    ctx.run(f"git push")
