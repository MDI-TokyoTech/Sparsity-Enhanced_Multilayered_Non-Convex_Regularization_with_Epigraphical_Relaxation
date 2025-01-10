% Original Author: Shunsuke Ono (ono@sp.ce.titech.ac.jp)
% Arranged: Akari Katsuma

function[Du] = Prox_DVTVnorm_vector(Du, gamma, wlumi)
% Du has 3 color channels and 2 dimentions( vertical and horizontal )

N = size(Du, 1) / 6;
L  = cat(2, Du(1+0*N:1*N), Du(1+1*N:2*N)); % [Dv, Dh]
Cb = cat(2, Du(1+2*N:3*N), Du(1+3*N:4*N));
Cr = cat(2, Du(1+4*N:5*N), Du(1+5*N:6*N));

onevec = ones(N, 1);
threshL = ((sqrt(sum(L.^2, 2))).^(-1))*gamma*wlumi;
threshC = ((sqrt(sum(Cb.^2, 2) + sum(Cr.^2, 2))).^(-1))*gamma;
threshL(threshL > 1) = 1;
threshC(threshC > 1) = 1;
coefL = (onevec - threshL);
coefC = (onevec - threshC);

% Du(1+0*N:1*N) = coefL.*Du(1+0*N:1*N);
% Du(1+1*N:2*N) = coefL.*Du(1+1*N:2*N);
% Du(1+2*N:3*N) = coefC.*Du(1+2*N:3*N);
% Du(1+3*N:4*N) = coefC.*Du(1+3*N:4*N);
% Du(1+4*N:5*N) = coefC.*Du(1+4*N:5*N);
% Du(1+5*N:6*N) = coefC.*Du(1+5*N:6*N);
for l = 1:2
    Du(1+(l-1)*N : (l+0)*N) = coefL.*Du(1+(l-1)*N : (l+0)*N);
    Du(1+(l+1)*N : (l+2)*N) = coefC.*Du(1+(l+1)*N : (l+2)*N);
    Du(1+(l+3)*N : (l+4)*N) = coefC.*Du(1+(l+3)*N : (l+4)*N);
end








