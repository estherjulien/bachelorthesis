from LocationTransportation.CCG.functions import master_problem, sub_problem, sub_problem_vertices
import numpy as np
import time


def ccg_function_vertices(env):
    # initialize
    start_time = time.time()
    z_set = [env.init_uncertainty]
    x_before_cut = dict()
    feas_gap_before_cut = dict()
    obj_init, _, x_init, _ = master_problem(env, env.vertices)
    # print("Instance {}: robust x: {}".format(env.inst_num, x_init))
    # print("Instance {}: robust obj: {}".format(env.inst_num, obj_init))

    it_time = dict()
    it = 1
    obj_new = 0
    while obj_init - obj_new > 1e-5:
        it_time[it-1] = time.time() - start_time
        # solve master problem
        obj_new, eta_new, x_new, y_new = master_problem(env, z_set)
        x_before_cut[it-1] = x_new
        # print("Instance {}: CCG obj = {}, #vertices = {}".format(env.instance_num, obj_new, len(cuts) - 1))
        # solve subproblem
        zeta, z_sol, _ = sub_problem_vertices(y_new, env)
        if zeta <= 1e-04:       # stop if robust
            break
        else:
            z_set.append(z_sol)

        feas_gap_before_cut[it-1] = zeta
        if obj_init - obj_new < 1e-5:   # or stop when objective of master problem with all vertices is smaller/equal (aka better) than new master problem objective
            break
        it += 1

    z_set_return = np.array(z_set[1:])

    # print("Instance {}: CCG x = {}".format(env.inst_num, x_sol_new))
    result_dict = {"x": x_new, "y": y_new, "eta_sol": eta_new, "vertices": z_set_return, "it_time": it_time, "x_before_cut": x_before_cut, "feas_gap": feas_gap_before_cut}
    return result_dict