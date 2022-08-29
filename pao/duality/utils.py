
class ComponentHasher(object):
    def __init__(self, var, bound):
        self.comp = var
        self.bound = bound

    def __eq__(self, other):
        if not isinstance(other, ComponentHasher):
            return False
        if self.comp is other.comp and self.bound is other.bound:
            return True
        else:
            return False

    def __hash__(self):
        return hash((id(self.comp), self.bound))

    def __repr__(self):
        first = str(self.comp)
        if self.bound is None:
            return first
        else:
            second = str(self.bound)
            res = str((first, second))
            res = res.replace("'", "")
            return res

    def __str__(self):
        return self.__repr__()
