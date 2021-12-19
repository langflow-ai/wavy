class A:
    def foo(self):
        "Class A foo"
        print('Class A foo')

class B(A):
    def foo2(self):
        "Class B foo2"
        print('Class B foo2')

    def foo(self):
        "Class B foo"
        return super().foo()

ibis = B()
ibis.foo()

# print(B.foo2.__doc__)
