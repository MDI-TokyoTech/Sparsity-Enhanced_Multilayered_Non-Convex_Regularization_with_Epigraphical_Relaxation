% Author: Shunsuke Ono (ono@sp.ce.titech.ac.jp)

function[Du] = Prox_DVTVnorm(Du, gamma, wlumi)

[v, h, ~, d] = size(Du);
onemat = ones(v, h);
threshL = ((sqrt(sum(Du(:,:,1,:).^2, 4))).^(-1))*gamma*wlumi;
threshC = ((sqrt(sum(sum(Du(:,:,2:3,:).^2, 4),3))).^(-1))*gamma;
threshL(threshL > 1) = 1;
threshC(threshC > 1) = 1;
coefL = (onemat - threshL);
coefC = (onemat - threshC);

for l = 1:d
    Du(:,:,1,l) = coefL.*Du(:,:,1,l);
    Du(:,:,2,l) = coefC.*Du(:,:,2,l);
    Du(:,:,3,l) = coefC.*Du(:,:,3,l);
end










