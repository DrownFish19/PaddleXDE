from .universal import ContainerMeta


class LEVY_AREA_APPROXIMATIONS(metaclass=ContainerMeta):  # noqa
    none = "none"  # Don't compute any Levy area approximation
    space_time = "space-time"  # Only compute an (exact) space-time Levy area
    davie = "davie"  # Compute Davie's approximation to Levy area
    foster = (
        "foster"  # Compute Foster's correction to Davie's approximation to Levy area
    )
