# gal-task

## Notes

- A lot of time was wasted using the CSV file method suggested.
  - I have dropped it in the current implementation as I don't understand the purpose

- Trying to automate the download from google was also a time drain, I spent far too long trying to implement this
- I have dropped this in the current implementation
- Phrase file should be passed in for simplicity


## Fixes to implement:

TODO: Do not require original model if CSV exitst
TODO: Fix all the mypy errors
TODO: Write intelligent encoding reader
TODO: Installed pandas - is this a good move?
TODO: Is ignoring words not in the model the correct approach
TODO: I assume the model limit is not picking up some standard words
TODO: Move the startup code into the embedding model initiation
TODO: Deal with bad input sentence

[ ] Working with the large flat file was not great on my machine. Using the CSV file method as specified made
my code unworkable and I have dropped it. I don't understand at all why I should be saving the model in a csv.


## Bonus Objectives

### Provide commentary and ideas how to structure and optimise the code in the future.
- Separate embedding model functionality into a local and remote model and use dependency injection to switch between them
- Use the ray datasets api for a bit more simplicity
- Use pydantic-settings to load settings from environment variables
- Enable passing in of arguments to overwrite default settings

### Process the data in chunks and in parallel by spinning up processes in parallel or by using a distributed processing framework (including the ones limited to a single node such as Polars)
- Done, but needs to be called from the cli

### Write some unit tests and test that could server as an entrypoint into the app (pytest preferably)
- Done but not to any great extent

### Structure code as pipeline step execution. Separate pipeline orchestration elements from data manipulation elements. Separate data manipulation from IO. Configure IO via configuration.
- Required minimal changes in my opinion

### Write an custom error handling decorator - raises an custom exception if code of the method raised an error. Include the previous errors in args. Annotate your methods with it instead of having try/except explicitly in your methods.
- Not done at all


### In Step 1  clean duplicates, outliers and stopwords from the phrases. In cases where the exact match is not found in the list of words from word2vec set, use the Levenshtein distance to find the closest similar word and use its vector instead
- Need to do as relatively easy

### Prepare a docker file building the image hosting the python app with all dependencies. If you are using a DB for data storage, have a separate image for the database and provide docker compose file to spawn the app.
- Not done, would like to implement fastapi and dockerise it



## Instructions for use

1.GoogleNews-vectors-negative300.bin.gz must be loaded to the ./data folder

For use of the cli, please use
```
uv run cli --help
```
