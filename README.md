## Setup

```bash
git clone https://github.com/abrar-sheikh/blog.git
cd blog
poetry install
```

## Running Projects
There are two ways to run the project
1. Directly on the notebook, this option is preferred from interactive development and debugging.
   1. Download and install VS Code and open the project in VS code. Install python, jupyter, Tensorboard plugins
   2. Open the appropreate notebook from the `./content` directory
   3. Select the kernal, specifically one identified by `poetry env list --full-path`
   4. Click on Run All
2. On command line, this option is preferred from running long running job in background
   1. From the root directory run `./nb_runner.sh content/<notebook_name_file>`


## Preview

```bash
quarto preview
```

## To publish blog

1. `git checkout main`
2. `git pull`
3. `quarto publish gh-pages`

