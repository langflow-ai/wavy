
dunder_methods = ['__add__', '__sub__', '__mul__', '__ge__', '__gt__', '__le__', '__lt__', '__pow__']

def _one_arg(list, value, __f):
    return [getattr(a, __f)(value) for a in list]

class Test:
    def __init__(self, blocks):
        # TODO: blocks must have increasing indexes, add warning and reindex
        self.list = blocks

    for dunder in dunder_methods:
        locals()[dunder] = lambda self, value, __f=dunder, *args, **kwargs: _one_arg(self.list, value, __f)


test = Test([1,2,3,4])
print(test.list)

print(test.__add__(1))
print(test.__mul__(2))
