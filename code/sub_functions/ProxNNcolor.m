% Author: Shunsuke Ono (ono@sp.ce.titech.ac.jp)

function y = ProxNNcolor(x, gamma)

[U1, S1, V1] = svd(x(:,:,1));
[U2, S2, V2] = svd(x(:,:,2));
[U3, S3, V3] = svd(x(:,:,3));
y1 = U1*max(S1-gamma,0)*V1';
y2 = U2*max(S2-gamma,0)*V2';
y3 = U3*max(S3-gamma,0)*V3';
y = cat( 3, y1, y2, y3 );