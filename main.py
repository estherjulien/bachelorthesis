from LocationTransportation.Environment.Env import Environment
from LocationTransportation.CCG.algorithm import ccg_function_vertices

C = 10
S = 5
gamma_percent = 0.9     # easiest
env = Environment(C, S, gamma_percent)

result = ccg_function_vertices(env)