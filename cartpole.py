from casadi import *
from single_shooting import single_shooting_transcription, single_shooting_nlp
import matplotlib.pyplot as plt

# optimization horizon time and number of discretization intervals
state_dim = 4
control_dim = 1

# Model variables
x1 = MX.sym("x1")  # cart position
x2 = MX.sym("x2")  # pole position
x3 = MX.sym("x3")  # cart velocity
x4 = MX.sym("x4")  # pole velocity
state = vertcat(x1, x2, x3, x4)
control = MX.sym("u")  # input force

# Model equations
gravity = 9.81
pole_length = 0.5
cart_mass = 10.0
pole_mass = 1.0
total_mass = cart_mass + pole_mass
dynamics_ode = vertcat(
    x3,
    x4,
    control
    + pole_mass
    * sin(x2)
    * (pole_length * x4**2 + gravity * cos(x2))
    / (cart_mass + pole_mass * sin(x2) ** 2),
    (
        -control * cos(x2)
        - pole_mass * pole_length * x4**2 * cos(x2) * sin(x2)
        - total_mass * gravity * sin(x2)
    )
    / (pole_length * cart_mass + pole_length * pole_mass * sin(x2) ** 2),
)

# Objective term
stage_cost = (
    1e0 * x1**2
    + 1e1 * (x2 - pi) ** 2
    + 1e-1 * x3**2
    + 1e-1 * x4**2
    + 1e-3 * control**2
)
final_cost = 1e0 * x1**2 + 1e1 * (x2 - pi) ** 2 + 1e-1 * x3**2 + 1e-1 * x4**2
Vf = Function("Vf", [state], [final_cost])


# constraints
ctrl_lb = -70.0
ctrl_ub = 70.0
path_constraints = Function(
    "g",
    [state],
    [0.0],
)
pc_ub = inf
pc_lb = -inf

# Formulate discrete time dynamics
disc_step = 0.05
down_sampling = 1
disc_dynamics_cost = single_shooting_transcription(
    state, control, dynamics_ode, stage_cost, disc_step, down_sampling
)

# mpc steps and time
sim_steps = 100
sim_time = sim_steps * disc_step
control_steps = 20


# initial state
optimal_states = [[0.0, 0.0, 0.0, 0.0]]

# initial solution guess

nlp_sol = np.zeros((control_steps, control_dim))

# variable to store optimal controls
optimal_controls = []

# tun mpc control
for k in range(sim_steps):
    # solve nlp with reduced horizon
    nlp_sol = single_shooting_nlp(
        optimal_states[-1],
        nlp_sol,
        disc_dynamics_cost,
        Vf,
        ctrl_lb,
        ctrl_ub,
        path_constraints,
        pc_lb,
        pc_ub,
        control_dim,
        control_steps,
    )

    # store first value of optimal control sequence
    optimal_controls = vcat([optimal_controls, nlp_sol[0]])

    # apply control
    Fk = disc_dynamics_cost(x0=optimal_states[-1], p=nlp_sol[0])
    optimal_states += [Fk["xf"].full()]

# plot solution
x1_opt = vcat([r[0] for r in optimal_states])
x2_opt = vcat([r[1] for r in optimal_states])
x3_opt = vcat([r[2] for r in optimal_states])
x4_opt = vcat([r[3] for r in optimal_states])
tgrid = [sim_time / sim_steps * k for k in range(sim_steps + 1)]
plt.figure(1)
plt.clf()
plt.plot(tgrid, x1_opt, "--")
plt.plot(tgrid, x2_opt, "-")
plt.xlabel("t")
plt.legend(["cart position", "pendulum angle"])
plt.grid()
plt.show()
plt.figure(2)
plt.step(tgrid, vertcat(DM.nan(1), optimal_controls))
plt.show()
