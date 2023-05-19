from casadi import *


def single_shooting_transcription(
    state_variable, control_variable, dynamics, objective, sampling_time, down_sampling
):
    # Formulate discrete time dynamics
    f = Function("f", [state_variable, control_variable], [dynamics, objective])
    X0 = MX.sym("X0", state_variable.shape[0])
    U = MX.sym("U", control_variable.shape[0])
    X = X0
    Q = 0

    # Fixed step Runge-Kutta 4 integrator
    for j in range(down_sampling):
        k1, k1_q = f(X, U)
        k2, k2_q = f(X + sampling_time / 2 * k1, U)
        k3, k3_q = f(X + sampling_time / 2 * k2, U)
        k4, k4_q = f(X + sampling_time * k3, U)
        X = X + sampling_time / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Q = Q + sampling_time / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)

    # discretized dynamics and stage cost
    F = Function("F", [X0, U], [X, Q], ["x0", "p"], ["xf", "qf"])
    return F


def single_shooting_nlp(
    initial_state,
    initial_guess,
    disc_dynamics_and_cost,
    final_cost,
    ctrl_lb,
    ctrl_ub,
    path_constraints,
    pc_lb,
    pc_ub,
    control_dim,
    control_steps,
):
    # Start with an empty NLP
    w = []  # NLP decision variable
    lbw = []  # decision variable lower bound
    ubw = []  # decision variable upper bound
    J = 0  # objective
    g = []  # constraints
    lbg = []  # constraints lower bound
    ubg = []  # constraints upper bound

    # Formulate the NLP
    Xk = initial_state
    for k in range(control_steps):
        # New NLP variable for the control
        Uk = MX.sym("U_" + str(k), control_dim)
        w += [Uk]
        lbw = vcat((lbw, ctrl_lb))
        ubw = vcat((ubw, ctrl_ub))

        # Integrate till the end of the interval
        Fk = disc_dynamics_and_cost(x0=Xk, p=Uk)
        Xk = Fk["xf"]
        J = J + Fk["qf"]

        # Add path constraints
        g += [path_constraints(Xk)]
        lbg += [pc_lb]
        ubg += [pc_ub]

    # add final cost
    J = J + final_cost(Xk)

    # Create an NLP solver
    prob = {"f": J, "x": vertcat(*w), "g": vertcat(*g)}

    # solver = nlpsol("solver", "sqpmethod", prob, {"qpsol": "qpoases"})
    solver = nlpsol("solver", "ipopt", prob)

    # Solve the NLP
    sol = solver(x0=initial_guess, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol["x"]
    return w_opt
