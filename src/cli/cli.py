import typer

app = typer.Typer()


@app.command()
def hello_world() -> None:
    print("hello world")


@app.command()
def hello(name: str) -> None:
    print(f"hello {name}")


if __name__ == "__main__":
    app()
