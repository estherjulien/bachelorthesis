import numpy as np
import gurobipy as gp
from gurobipy import GRB
import copy


def master_problem(z_set, env):
    S = env.S
    C = env.C
    custs = env.customers
    facs = env.facilities
    N = len(z_set)

    # Model
    mp = gp.Model("Master Problem")
    mp.Params.OutputFlag = 0
    # variables
    eta = mp.addVar(lb=0, vtype=GRB.CONTINUOUS)
    x = mp.addVars(S, name="x", vtype=GRB.BINARY)
    p = mp.addVars(S, lb=0, name="p", vtype=GRB.CONTINUOUS)
    y_index = [(s, c) for s in np.arange(S) for c in np.arange(C)]
    y = dict()
    for i in np.arange(N):
        y[i] = mp.addVars(y_index, lb=0, name="y_{}".format(i), vtype=GRB.CONTINUOUS)

    # objective function
    mp.setObjective(gp.quicksum(facs[s].var_cost * p[s] + facs[s].fix_cost * x[s] for s in np.arange(S)) + eta, GRB.MINIMIZE)

    # first-stage constraint            CAN CAUSE INFEASIBILITY
    mp.addConstrs(p[s] <= facs[s].capacity * x[s] for s in np.arange(S))

    # combination constraint
    for i in np.arange(N):
        mp.addConstrs(gp.quicksum(y[i][s, c] for c in np.arange(C)) <= p[s] for s in np.arange(S))

    # second-stage objective constraints
    for i in np.arange(N):
        mp.addConstr(eta >= gp.quicksum(env.trans_cost[s, c]*y[i][s, c] for s in np.arange(S) for c in np.arange(C)))

    # uncertain
    for i, z in enumerate(z_set):
            mp.addConstrs((gp.quicksum(y[i][s, c] for s in np.arange(S)) >= (custs[c].demand + z[c]*custs[c].delta)) for c in np.arange(C))

    # solve
    mp.optimize()
    try:
        x_sol = np.array([var.X for i, var in x.items()])
    except:
        print("Instance {}: INFEASIBLE MASTER PROBLEM".format(env.inst_num))
    p_sol = np.array([var.X for i, var in p.items()])
    y_sol = dict()
    for j in np.arange(N):
        y_sol[j] = np.array([var.X for i, var in y[j].items()]).reshape(S, C)
    eta_sol = eta.X

    theta = mp.objVal
    return theta, eta_sol, {"p": p_sol, "x": x_sol}, y_sol


def sub_problem(y_input, env):
    S = env.S
    C = env.C
    custs = env.customers
    N = len(y_input)

    # Model
    ic = gp.Model("Infeasible check")
    ic.Params.OutputFlag = 0
    ic.Params.IntFeasTol = 1e-9

    # variables
    z = ic.addVars(C, lb=0, ub=env.z_hat, vtype=GRB.CONTINUOUS)
    b_index = [(i, c) for i in np.arange(N) for c in np.arange(C)]
    b = ic.addVars(b_index, vtype=GRB.BINARY)
    zeta = ic.addVar(lb=-env.capacity_full, vtype=GRB.CONTINUOUS)        # sum ysc over s is maximum the capacity!

    # objective
    ic.setObjective(zeta, GRB.MAXIMIZE)     # initially maximize

    # constraint binary variable
    ic.addConstrs(gp.quicksum(b[i, c] for c in np.arange(C)) == 1 for i in np.arange(N))

    # constraints model
    for i, y in y_input.items():
        ic.addConstrs(zeta + env.bigM*b[i, c] <= -sum(y[s, c] for s in np.arange(S)) + custs[c].demand + z[c]*custs[c].delta + env.bigM
                      for c in np.arange(C))

    # constraint uncertainty set
    ic.addConstr(gp.quicksum(z[c] for c in np.arange(C)) <= env.gamma)

    # solve
    ic.optimize()
    z_sol = np.array([var.X for i, var in z.items()])
    zeta_sol = zeta.X
    b_sol = np.array([var.X for i, var in b.items()])
    return zeta_sol, z_sol, b_sol


def sub_problem_vertices(y_input, env):
    S = env.S
    C = env.C
    custs = env.customers
    N = len(y_input)
    z = env.vertices

    # Model
    ic = gp.Model("Infeasible check")
    ic.Params.OutputFlag = 0
    ic.Params.IntFeasTol = 1e-9

    # variables
    z_bin = ic.addVars(len(z), vtype=GRB.BINARY)
    b_index = [(i, c) for i in np.arange(N) for c in np.arange(C)]
    b = ic.addVars(b_index, vtype=GRB.BINARY)
    zeta = ic.addVar(lb=-env.capacity_full, vtype=GRB.CONTINUOUS)  # sum ysc over s is maximum the capacity!

    # objective
    ic.setObjective(zeta, GRB.MAXIMIZE)  # initially maximize

    # constraints model
    for i, y in y_input.items():
        ic.addConstrs(zeta + env.bigM * b[i, c] <= -sum(y[s, c] for s in np.arange(S)) + custs[c].demand +
                      gp.quicksum(z[l][c]*z_bin[l] for l in np.arange(len(z))) * custs[c].delta + env.bigM
                      for c in np.arange(C))

    # constraint binary variable
    ic.addConstrs(gp.quicksum(b[i, c] for c in np.arange(C)) == 1 for i in np.arange(N))

    # constraint vertex variable
    ic.addConstr(gp.quicksum(z_bin[l] for l in np.arange(len(z))) == 1)

    # solve
    ic.optimize()
    z_bin_sol = np.where(np.array([var.X for i, var in z_bin.items()]))
    z_sol = z[z_bin_sol]
    zeta_sol = zeta.X
    b_sol = np.array([var.X for i, var in b.items()])
    return zeta_sol, z_sol, b_sol