import ray

# Initialize Ray
ray.init()
ray.init(ignore_reinit_error=True, num_cpus=4)
print("success")


@ray.remote
def process_item(item):
    processed_item = item + 1.5
    return processed_item


def main():
    # Example list of items to process
    items_to_process = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Create a list to store references to remote tasks
    tasks = []

    # Submit tasks for parallel execution
    for item in items_to_process:
        task = process_item.remote(item)
        tasks.append(task)

    # Get results from the tasks
    results = ray.get(tasks)
    print("Processed results:", results)


if __name__ == "__main__":
    main()