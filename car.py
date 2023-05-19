from casadi import *
from single_shooting import single_shooting_nlp, single_shooting_transcription
import matplotlib.pyplot as plt

# state and control vectors dimensions
state_dim = 6
control_dim = 2

# Model variables
x1 = MX.sym("x1")  # x - position on x-axis
x2 = MX.sym("x2")  # y - position on y-axis
x3 = MX.sym("x3")  # phi - yaw angle
x4 = MX.sym("x4")  # u - velocity on x-axis
x5 = MX.sym("x5")  # v - velocity on y-axis
x6 = MX.sym("x6")  # omega - yaw rate
state = vertcat(x1, x2, x3, x4, x5, x6)
u1 = MX.sym("u1")  # a - input acceleration
u2 = MX.sym("u2")  # delta - input steering
control = vertcat(u1, u2)

# model parameters
m = 1412.0
lf = 1.06
lr = 1.85
kf = -128916.0
kr = -85944.0
Ts = 0.1
L = lf * kf - lr * kr
Iz = 1536.7

# ode
dynamics_ode = vertcat(
    x4 * cos(x3) - x5 * sin(x3),
    x4 * sin(x3) + x5 * cos(x3),
    x6,
    u1 + x5 * x6 - 1 / m * kf * ((x5 + lf * x6) / x4 - u2) * sin(u2),
    -x4 * x6 + 1 / m * (kf * ((x5 + lf * x6) / x4 - u2) + kr * (x5 - lr * x6) / x4),
    1 / Iz * (lf * kf * ((x5 + lf * x6) / x4 - u2) - lr * kr * (x5 - lr * x6) / x4),
)

# stage and final cost
stage_cost = 0.5 * x2**2 + 0.5 * (x4 - 8) ** 2 + u1**2 + u2**2
final_cost = 0.5 * x2**2 + 0.5 * (x4 - 8) ** 2
Vf = Function("Vf", [state], [final_cost])


# constraints
# controls lower and upper bounds
ctrl_lb = [-3, -0.6]
ctrl_ub = [1.5, 0.6]

# ellipse obstacle
ea = 5.0
eb = 2.5
cx = 15.0
cy = -1.0
path_constraints = Function(
    "g",
    [state],
    [1 - (1.0 / ea**2 * (x1 - cx) ** 2 + 1.0 / eb**2 * (x2 - cy) ** 2)],
)
pc_ub = 0.0
pc_lb = -inf

# discretize dynamics and cost
disc_step = 0.05
down_sampling = 1
disc_dynamics_cost = single_shooting_transcription(
    state, control, dynamics_ode, stage_cost, disc_step, down_sampling
)

# mpc steps and time
sim_steps = 120
sim_time = sim_steps * disc_step
control_steps = 20

# initial state
optimal_states = [[0.0, 0.0, 0.0, 5.0, 0.0, 0.0]]

# initialize solution guess
nlp_sol = np.zeros((control_steps * control_dim, 1))

# variable to stroe optimal controls
optimal_controls = []

# simulate sistem with optimal controls
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
    optimal_controls = vcat([optimal_controls, nlp_sol[:control_dim]])

    # apply control
    Fk = disc_dynamics_cost(x0=optimal_states[-1], p=nlp_sol[:control_dim])
    optimal_states += [Fk["xf"].full()]

# plot solution
x1_opt = vcat([r[0] for r in optimal_states])
x2_opt = vcat([r[1] for r in optimal_states])
x3_opt = vcat([r[2] for r in optimal_states])
x4_opt = vcat([r[3] for r in optimal_states])
x5_opt = vcat([r[3] for r in optimal_states])
x6_opt = vcat([r[3] for r in optimal_states])

t = linspace(0, 2 * pi, 150)
plt.plot(cx + ea * cos(t), cy + eb * sin(t), color="red")
plt.plot(x1_opt, x2_opt)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
tgrid = [sim_time / sim_steps * k for k in range(sim_steps + 1)]
plt.plot(tgrid, x4_opt)
plt.xlabel("t")
plt.legend(["x velocity"])
plt.show()
optimal_controls = optimal_controls.reshape((control_dim, -1)).T
plt.step(tgrid, vertcat(DM.nan(1), optimal_controls[:, 0]))
plt.xlabel("t")
plt.legend(["acceleration"])
plt.show()
plt.step(tgrid, vertcat(DM.nan(1), optimal_controls[:, 1]))
plt.xlabel("t")
plt.legend(["steering"])
plt.show()
