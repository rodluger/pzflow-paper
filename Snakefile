rule train_two_moons_flow:
    output:
        "src/data/two_moons_flow.pzflow.pkl"
    script:
        "src/scripts/train_two_moons.py"
