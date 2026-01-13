from enum import Enum

class EnvEnum(str, Enum):
    LinearSystem = "LinearSystem-v0"
    FluidFlow = "FluidFlow-v0"
    Lorenz = "Lorenz-v0"
    DoubleWell = "DoubleWell-v0"