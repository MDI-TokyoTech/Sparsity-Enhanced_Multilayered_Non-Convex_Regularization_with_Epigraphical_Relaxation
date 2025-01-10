% Author: Shunsuke Ono (ono@sp.ce.titech.ac.jp)

function[Du] = Prox_DVTVnorm_Jcb(Du, gamma, wlumi)

[v, h] = size(Du);
onemat = ones(v, h);
threshL = ((sqrt(sum(Du(:,:,1,:,:,:).^2, 3))).^(-1))*gamma*wlumi;
threshC = ((sqrt(sum(Du(:,:,2:3,:,:,:).^2, 3))).^(-1))*gamma;

% threshC = ((sqrt(sum(sum(Du(:,:,2:3,:,:,:).^2, 4),3))).^(-1))*gamma;
threshL(threshL > 1) = 1;
threshC(threshC > 1) = 1;
coefL = (1 - threshL);
coefC = (1 - threshC);

Du(:,:,1,:,:,:) = coefL.*Du(:,:,1,:,:,:);
Du(:,:,2,:,:,:) = coefC.*Du(:,:,2,:,:,:);
Du(:,:,3,:,:,:) = coefC.*Du(:,:,3,:,:,:);










