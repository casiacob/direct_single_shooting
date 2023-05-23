from casadi import *
from single_shooting import single_shooting_transcription, single_shooting_nlp
import matplotlib.pyplot as plt


state_dim = 2
control_dim = 1

# Model variables
x1 = MX.sym("x1")
x2 = MX.sym("x2")
state = vertcat(x1, x2)
control = MX.sym("u")

# Model equations
gravity = 9.81
length = 1.0
mass = 1.0
damping = 1e-3
dynamics_ode = vertcat(x2, -gravity / length * sin(x1) + (control - damping * x2) / (mass * length))

# Objective term
lc = 2e0 * (fmod(x1, 2*pi) - pi) ** 2 + 1e-1 * x2**2 + 1e-3 * control**2
stage_cost = Function("l", [state, control], [0.5 * lc])
Vf = 2e0 * (fmod(x1, 2*pi) - pi) ** 2 + 1e-1 * x2**2
final_cost = Function("Vf", [state], [0.5 * Vf])

# constraints
ctrl_lb = -5.0
ctrl_ub = 5.0
path_constraints = Function(
    "g",
    [state],
    [0.0],
)
pc_ub = inf
pc_lb = -inf


# Formulate discrete time dynamics
disc_step = 0.01
down_sampling = 5
disc_dynamics = single_shooting_transcription(
    state, control, dynamics_ode, disc_step, down_sampling
)


# mpc steps and time
sim_steps = 100
sim_time = sim_steps * disc_step
horizon = 40

# intial state
optimal_states = [[fmod(0.01, 2*pi), -0.01]]

# initial solution guess
nlp_sol = np.zeros((horizon, control_dim))

# variable to store optimal controls
optimal_controls = []

for k in range(sim_steps):
    # solve nlp
    nlp_sol = single_shooting_nlp(
        optimal_states[-1],
        nlp_sol,
        disc_dynamics,
        stage_cost,
        final_cost,
        disc_step,
        ctrl_lb,
        ctrl_ub,
        path_constraints,
        pc_lb,
        pc_ub,
        control_dim,
        horizon,
    )

    # store first value of optimal controls sequence
    optimal_controls = vcat([optimal_controls, nlp_sol[0]])

    # apply control
    Fk = disc_dynamics(x0=optimal_states[-1], p=nlp_sol[0])
    optimal_states += [Fk["xf"].full()]

# plot solution
x1_opt = vcat([r[0] for r in optimal_states])
x2_opt = vcat([r[1] for r in optimal_states])
tgrid = [sim_time / sim_steps * k for k in range(sim_steps + 1)]
plt.clf()
plt.plot(tgrid, x1_opt, "-")
plt.plot(tgrid, x2_opt, "--")
plt.xlabel("t")
plt.legend(["angle", "ang vel"])
plt.grid()
plt.show()
plt.step(tgrid, vertcat(DM.nan(1), optimal_controls))
plt.xlabel("t")
plt.legend(["torque"])
plt.show()
