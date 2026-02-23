import json


def load_movies():
    with open("./data/movies.json", "r") as f:
        return json.load(f)["movies"]
