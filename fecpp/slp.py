import numpy as np

from fecpp.varfilter import VarFilter


class SeaLevelPressure(VarFilter):
    def should_filter(self, in_var):
        if in_var.name == 'PSFC':
            self.name = 'SLP'
            self.standard_name = "air_pressure_at_mean_sea_level"
            self.long_name = "Air pressure reduced to mean sea level"
            self.units = "Pa"

            return True
        else:
            return False

    def filtered(self, index):
        temp = self.original.group().variables['T2D'][index]
        mixing = self.original.group().variables['Q2D'][index]
        height = self.height[index]

        press = self.original.__getitem__(index)

        return slp(temp, mixing, height, press)


def slp(temp, mixing, height, press):
    g0 = 9.80665
    Rd = 287.058
    epsilon = 0.622

    Tv = temp*(1 + (mixing/epsilon))/(1 + mixing)
    H = Rd * Tv / g0

    press_sl = press / np.exp(-height / H)

    return press_sl


if __name__ == '__main__':
    from math import isclose
    assert isclose(slp(temp=302.71, mixing=0.019, height=219.456, press=988.69), 1013.2, rel_tol=0.01)
