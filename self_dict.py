class A(object):
    def __init__(self):
        self.a = 0
        self.b = 0

    def to_dict(self):
        """
        Converts the instance variables to a dictionary
        """
        return self.__dict__
    
a = A()
print(a.to_dict())

a = [0, None, 0]
print(a)