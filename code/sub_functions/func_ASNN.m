function output = func_ASNN(u_opt)
% ASNNの値を計算する
%   入力：低ランク2次元イメージ

[rows, ~] = size(u_opt);

Wc = 1/sqrt(rows)*cos(2*pi/rows*((0:rows-1)')*(0:rows-1));
Ws = 1/sqrt(rows)*sin(2*pi/rows*((0:rows-1)')*(0:rows-1));
temp = Wc'*Wc  + (-Ws)'*(-Ws);
scale = 1/sqrt(temp(1,1));
Wc = scale*Wc;
Ws = scale*Ws;
T = @(z) cat(3, Wc*z, -Ws*z);

L2norm = sqrt( sum( T(u_opt).^2 , 3 ) );
output = sum( svd(L2norm) );
end

