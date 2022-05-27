rule train_two_moons_flow:
    output:
        "src/data/two_moons_flow.pzflow.pkl"
    cache:
        True
    script:
        "src/scripts/train_two_moons_flow.py"

rule train_main_galaxy_flow:
    output:
        "src/data/main_galaxy_flow.pzflow.pkl"
        "src/data/main_galaxy_flow_losses.npy"
    cache:
        True
    script:
        "src/scripts/train_main_galaxy_flow.py"

rule train_conditional_galaxy_flow:
    output:
        "src/data/conditional_galaxy_flow.pzflow.pkl"
        "src/data/conditional_galaxy_flow_losses.npy"
    cache:
        True
    script:
        "src/scripts/train_conditional_galaxy_flow.py"
