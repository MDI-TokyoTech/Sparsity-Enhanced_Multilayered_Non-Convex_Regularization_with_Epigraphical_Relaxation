% Author: Shunsuke Ono (ono@isl.titech.ac.jp)

function[Du] = Prox_TVnorm(Du, gamma)

[ v, h, ~, ~, ~, ~ ] = size(Du);
onemat = ones(v, h);
thresh = (sqrt(sum(Du.^2, 3)).^(-1))*gamma;
thresh(thresh > 1) = 1;
coef = repmat((onemat - thresh),1,1,size(Du,3));
Du = coef.*Du;











