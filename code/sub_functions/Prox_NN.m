% Author: Shunsuke Ono (ono@sp.ce.titech.ac.jp)

function y = Prox_NN(x, gamma)

[U, S, V] = svd(x);
y = U*max(S-gamma,0)*V';