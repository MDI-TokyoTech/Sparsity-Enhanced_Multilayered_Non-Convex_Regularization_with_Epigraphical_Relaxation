% Author: Shunsuke Ono (ono@sp.ce.titech.ac.jp)

% function[Du] = Prox_L12norm(Du, gamma, mode, argin)
function[Du] = Prox_L12norm(Du, gamma)

[v, h, c, d] = size(Du);
onemat = ones(v, h);
thresh = ((sqrt(sum(sum(Du.^2, 4),3))).^(-1))*gamma;
thresh(thresh > 1) = 1;
coef = (onemat - thresh);
Du = repmat(coef,1,1,c,d).*Du;
