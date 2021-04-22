import numpy as np
import pypoman as pm


class Customer:
    def __init__(self):
        self.x = np.random.uniform(0, 10)
        self.y = np.random.uniform(0, 10)
        self.demand = np.random.uniform(10, 500)
        self.delta = np.random.uniform(0.1, 0.5)*self.demand


class Facility:
    def __init__(self, capacity):
        self.fix_cost = np.random.uniform(1, 10)
        self.var_cost = np.random.uniform(0.1, 1)
        self.capacity = capacity


class Environment:
    def __init__(self, C, S, gamma_percent, inst_num=0):
        self.z_hat = 1
        self.C = C
        self.S = S
        self.gamma_percent = gamma_percent
        self.gamma = gamma_percent * self.C
        self.inst_num = inst_num

        # make customers
        customers = dict()
        for i in np.arange(self.C):
            customers[i] = Customer()
        self.customers = customers

        self.capacity_full = sum([self.customers[c].demand + self.z_hat * self.customers[c].delta for c in np.arange(C)])

        # make facilities
        facilities = dict()
        for s in np.arange(self.S):
            facilities[s] = Facility(self.capacity_full)
        self.facilities = facilities

        # make distances between facilities and customers
        trans_cost = dict()
        for s, facility in facilities.items():
            for c, customer in customers.items():
                trans_cost[s, c] = np.random.uniform(0, 10)
        self.trans_cost = trans_cost

        # bigM for subproblem
        self.bigM = max([self.customers[c].demand + self.z_hat * self.customers[c].delta for c in np.arange(C)])
        # upper bound
        self.upper_bound = self.S * self.C * max([trans_cost[s, c] for s in np.arange(S) for c in np.arange(C)]) * self.bigM + \
                           self.capacity_full * max([self.facilities[s].var_cost for s in np.arange(S)]) + \
                           sum([self.facilities[s].fix_cost for s in np.arange(S)])
        # initial uncertainty
        self.init_uncertainty = np.zeros(C, dtype=np.float)
        # vertices of uncertainty set
        self.vertices = vertex_fun(self)


def vertex_fun(env):
    S = env.S
    C = env.C
    custs = env.customers
    num_consts = 1+2*C

    # pypoman (pm) for vertices
    # uncertainty set is C-dimensional
    A = np.concatenate((np.ones([1, C]), np.eye(C), -1*np.eye(C)), axis=0)
    b = np.concatenate((np.array([env.gamma]).reshape(1, 1), env.z_hat * np.ones([1, C]), np.zeros([1, C])), axis=1).reshape(num_consts)
    vertices_array = np.array(pm.compute_polytope_vertices(A, b))
    max_sum_vertices = max(np.sum(vertices_array, axis=1))
    needed_vertices = vertices_array[np.where(np.sum(vertices_array, axis=1) >= max_sum_vertices)]
    return needed_vertices

