# Blog Project

## Setup

Clone the repository and install dependencies using Poetry:

```bash
git clone https://github.com/abrar-sheikh/blog.git
cd blog
poetry install
```

## Running the Project

You can run the project in two ways:

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

### Preview

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
