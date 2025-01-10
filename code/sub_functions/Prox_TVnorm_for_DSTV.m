function [ x_new ] = Prox_TVnorm_for_DSTV(x, gamma)

x(:,:,2,:,:,:) = sqrt( sum( x(:,:,2:3,:,:,:).^2 , 3 ) );
x_new = Prox_TVnorm(x, gamma);