from run_experiments import start_experiments

results = start_experiments(
    runs=50,
    seed=100,
    import_stacks=8,
    export_stacks=8,
    new_containers=100,
    initial_containers=50,
    populate_initial=True,
    save_results=False,
    verbose=True,
)

print(results["summary"])