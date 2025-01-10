function[u] = Proj_RangeConstraint(u, range)

u(u < range(1)) = range(1);
u(u > range(2)) = range(2);