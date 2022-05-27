rule train_two_moons_flow:
    output:
        "src/data/two_moons_flow.pzflow.pkl"
    cache:
        True
    script:
        "src/scripts/train_two_moons_flow.py"

rule train_main_galaxy_flow:
    output:
        directory("src/data/main_galaxy_flow")
    cache:
        True
    script:
        "src/scripts/train_main_galaxy_flow.py"
