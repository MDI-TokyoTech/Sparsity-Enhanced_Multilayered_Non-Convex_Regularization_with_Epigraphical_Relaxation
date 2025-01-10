% Author: Shunsuke Ono (ono@sp.ce.titech.ac.jp)

function[u] = ProjL2ball2(u, f, epsilon)
for i = 1:2
radius = sqrt(sum(sum(sum((u(:,:,i) - f(:,:,i)).^2))));
if radius > epsilon
    u(:,:,i) = f(:,:,i) + (epsilon/radius)*(u(:,:,i) - f(:,:,i));
end
end