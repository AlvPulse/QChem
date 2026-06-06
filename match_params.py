def calculate_classical_hidden_dim(target_params, out_dim=617, in_dim=64):
    """
    Given a target parameter count from a Quantum model, solve for the closest
    hidden_dim for a simple Classical MLP (like Level 1/2) that uses
    Linear(in_dim, hidden_dim) -> Linear(hidden_dim, out_dim).

    Formula:
    Params = (in_dim * hidden_dim + hidden_dim) + (hidden_dim * out_dim + out_dim)
    Params = hidden_dim * (in_dim + 1 + out_dim) + out_dim
    hidden_dim = (Params - out_dim) / (in_dim + out_dim + 1)
    """
    hidden_dim = (target_params - out_dim) / (in_dim + out_dim + 1)
    return max(1, int(round(hidden_dim)))

print(calculate_classical_hidden_dim(50000, 617, 64))
