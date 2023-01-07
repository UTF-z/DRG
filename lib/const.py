class ImmutableClass(type):

    def __call__(cls, *args, **kwargs):
        raise AttributeError("Cannot instantiate this class")

    def __setattr__(cls, name, value):
        raise AttributeError("Cannot modify immutable class")

    def __delattr__(cls, name):
        raise AttributeError("Cannot delete immutable class")


class Queries(metaclass=ImmutableClass):
    IMG = 'img'
    LABEL = 'label'
    RES = 'res'
    ACC = 'acc'
    LOSS = 'loss'