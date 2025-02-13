# Blog

Access the blog at [abrarsheikh.dev](https://abrarsheikh.dev).

## Setup

Setup pyenv

mac
```bash
brew update  
brew install pyenv 
```

linux
```bash
curl https://pyenv.run | bash 
```

Then, add the following lines to your shell configuration file (~/.bashrc, ~/.zshrc, or ~/.bash_profile):
```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
```

Reload the shell:
```bash
exec "$SHELL"
```

install py3.11
```bash
pyenv install 3.11
```

Clone the repository and install dependencies:

```bash
pip install poetry
git clone https://github.com/abrar-sheikh/blog.git
cd blog
pyenv local 3.11
poetry install
```

## Running the Project

There are multiple projects in this repo, you can run a project in two ways:

### Interactive Development (Recommended)

Run the project directly in a Jupyter Notebook for easier debugging and experimentation:

1. Install VS Code and open the project.
2. Install the Python, Jupyter, and TensorBoard extensions.
3. Open the relevant notebook from the ./content directory.
4. Select the appropriate kernel using:
	```bash
	poetry env list --full-path
	```
5. Select the right kernal in VS code based on what you see in previous step
6. Click Run All to execute the notebook.

### Command-Line Execution (For Background Jobs)

Run long-running notebooks in the background via the command line:

```bash
./nb_runner.sh content/<notebook_filename>
```

### Preview blog

To preview the blog locally:

```bash
quarto preview
```

### Publishing the Blog

Deploy updates to GitHub Pages with:

```bash
git checkout main
git pull
quarto publish gh-pages
```
