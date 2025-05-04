import os




def display_thesis_structure(
    path, prefix="", max_depth=1, current_depth=0
):
    # Skip hidden, cache, environment directories
    if current_depth > max_depth:
        return

    items = sorted([
        item for item in os.listdir(path)
        if os.path.isdir(os.path.join(path, item))
        and not item.startswith(".")
        and not item.endswith("_env")
        and item != "__pycache__"
    ])

    pointers = ["├── ", "└── "]

    for index, item in enumerate(items):
        item_path = os.path.join(path, item)
        pointer = pointers[1] if index == len(items) - 1 else pointers[0]
        print(f"{prefix}{pointer}{item}")

        # Don't recurse into hyperparameter_tuning_results, but still show it
        if item == "hyperparameter_tuning_results":
            continue

        extension = "    " if pointer == pointers[1] else "│   "
        display_thesis_structure(
            item_path, prefix + extension, max_depth, current_depth + 1
        )

thesis_directory = "../"
display_thesis_structure(thesis_directory)
