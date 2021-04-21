from abc import ABC, abstractmethod


class VarFilter(ABC):
    def __init__(self, in_var, **kwargs):
        self.original = in_var
        self.name = in_var.name
        self.will_filter = self.should_filter(in_var)
        self.__dict__.update(kwargs)

    @abstractmethod
    def should_filter(self, in_var):
        raise RuntimeError

    @abstractmethod
    def filtered(self, index):
        raise RuntimeError

    def __getattr__(self, item):
        return self.original.__getattribute__(item)

    def __getitem__(self, index):
        if self.will_filter:
            return self.filtered(index)
        else:
            return self.original.__getitem__(index)

    def __setitem__(self, key, value):
        raise RuntimeError("VarFilter is a read-only view on a netCDF4.Variable")
