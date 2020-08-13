

solver_registry = {}

def register_solver(name, cls):
    solver_registry[name] = cls


def BilevelSolver(name):
    assert (name in solver_registry), "Unknown solver '%s' specified" % name
    return solver_registry[name]()
