# gal-task

## Introduction

I made some strange decisions, and my codebase spiralled.








## Bonus Objectives

### Provide commentary and ideas how to structure and optimise the code in the future.
Overall I am pretty happy with the codebase.
- Separate embedding model functionality into a local and remote model and use dependency injection to switch between them
- Enable passing in of arguments to overwrite default settings
- Better documentation

### Process the data in chunks and in parallel by spinning up processes in parallel or by using a distributed processing framework (including the ones limited to a single node such as Polars)
- Completed, polars provides out of memory streaming functionality under the covers

### Write some unit tests and test that could server as an entrypoint into the app (pytest preferably)
- Done a couple, shows what I would do

### Structure code as pipeline step execution. Separate pipeline orchestration elements from data manipulation elements. Separate data manipulation from IO. Configure IO via configuration.
- Completed to a large extent.

### Write an custom error handling decorator - raises an custom exception if code of the method raised an error. Include the previous errors in args. Annotate your methods with it instead of having try/except explicitly in your methods.
- Happy to do, but will leave this


### In Step 1  clean duplicates, outliers and stopwords from the phrases. In cases where the exact match is not found in the list of words from word2vec set, use the Levenshtein distance to find the closest similar word and use its vector instead
- Happy to do, but will leave this. I would use the Levenshtein library to calculate

### Prepare a docker file building the image hosting the python app with all dependencies. If you are using a DB for data storage, have a separate image for the database and provide docker compose file to spawn the app.
- Not done, would like



## Instructions for use

### Step 1
Installation requires uv - which can be installed from here
https://github.com/astral-sh/uv

### Step 2
To install all the required libraries, run `uv sync` in the folder

### Step 3
The CLI should be available at this time.


## Project structure

Input data goes in the input_data folder
The processes use the working_data folder to store intermediate data
The output data is stored in the output_data folder


### CLI Commands
Get cross similarities for any given model
    uv run cli get-cross-similarities-for-phrase <phrase_filename> <model_filename>

Get most similar phrases for any given phrase
    uv run cli get-most-similar-phrases-for-phrase <phrase_filename> <model_filename>
