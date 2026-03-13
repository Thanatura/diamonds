
truc = {"a": 1, "b": 2, "c": 3}

def show_object(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

show_object(truc)