function PDu = Prox_STVnorm_vector(PDu, gamma, blocksize)
% input vector in R^{patch_num x 1}
% [Dv_1,R, Dv_1,G, Dv_1,B, Dh_1,R, Dh_1,G, Dh_1,B, Dv_2,R, Dv_2,G,...]^T

% matrix of n-th patch in R^{p x 2}
% [Dv_n,R, Dh_n,R]
% [Dv_n,G, Dh_n,G]
% [Dv_n,B, Dh_n,B]

c = 3; % color channels
p = prod(blocksize)*c*2; % patch size
len = prod(blocksize)*c; % D size

for i = 1:p:size(PDu, 1)
    index_v = i:i+len-1;
    index_h = i+len:i+p-1;
    M = cat(2, PDu(index_v), PDu(index_h));
    [U, S, V] = svd(M,0);
    Sthre = diag(max(0, diag(S) - gamma));
    M2 = U*Sthre*V';
    PDu(i:i+p-1) = cat(1, M2(:,1), M2(:,2));
end