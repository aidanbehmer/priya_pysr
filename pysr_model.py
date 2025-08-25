from pysr import PySRRegressor

def create_model(niterations=20, maxsize=20, maxdepth=10, random_state=42):
    return PySRRegressor(
        model_selection="best",
        niterations=niterations,
        binary_operators=["+", "*", "-", "/", "^"],
        unary_operators=[
            "sin", "cos", "exp", "log", "square", "sqrt", "inv(x) = 1/x"
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        loss="loss(x, y) = (x - y)^2",
        maxsize=maxsize,
        maxdepth=maxdepth,
        verbosity=1,
        random_state=random_state,
    )
