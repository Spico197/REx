class Instance(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return "Instance(" + str(self.__dict__) + ")"
