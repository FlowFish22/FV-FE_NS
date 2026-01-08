print("simple code")


def scheme(u_0):
    """Scheme that depends only on the initial condition"""
    return 0


# Running the scheme for different initial condition
for u_0 in [lambda x: np.cos(x), lambda x: np.sin(x)]:
    scheme(u_0)
