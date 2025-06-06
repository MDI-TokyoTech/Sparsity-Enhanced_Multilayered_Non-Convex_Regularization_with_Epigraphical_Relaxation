function [ xp , zp ] = Proj_Epigraph_L2_for_DVTV_GPU_chro(x, z, w)
% L2 projection
% x = (rows, cols, color, vh)
[rows, cols, ~, ~] = size(x);

% Create difference vectors and Set side by side
xv = cat( 2, reshape( x(:,:,1,1), [ numel(z), 1 ] ), reshape( x(:,:,1,2), [ numel(z), 1] ), reshape( x(:,:,2,1), [ numel(z), 1 ] ), reshape( x(:,:,2,2), [ numel(z), 1] ) );
zv = reshape( z ,[ numel(z) , 1]  );
xpv = zeros( size(xv) );
zpv = zeros( size(zv) );

% xv = gpuArray(xv);
% zv = gpuArray(zv);
% xpv = zeros( size(xv),"gpuArray" );
% zpv = zeros( size(zv),"gpuArray" );

% Check each element if it is in the epigraph
xL2 = sqrt(sum(xv.^2, 2));
ind1 = ( xL2 < -w*(zv+eps) );
ind2 = ( xL2 <= (zv+eps)/w );
ind3 = ~or( ind1 , ind2 );

% Cast to 0
xpv(ind1==1,:) = 0;
zpv(ind1==1) = 0;

% If it is in the epigraph, it stays.
xpv(ind2==1,:) = xv(ind2==1,:);
zpv(ind2==1) = zv(ind2==1);

% Cast to nearest point in the epigraph
% xpv(ind3==1,:) = 1/(1+w^2) * ( 1 + w*zv(ind3==1,:)./sqrt(sum(xv(ind3==1,:).^2,2)) ).*xv(ind3==1,:);
% zpv(ind3==1) = 1/(1+w^2) * ( 1 + w*zv(ind3==1,:)./sqrt(sum(xv(ind3==1,:).^2,2)) )*w.*sqrt(sum(xv(ind3==1,:).^2,2));
xpv(ind3==1,:) = 1/(1+w^2) * ( 1 + w*zv(ind3==1,:)./xL2(ind3==1) ).*xv(ind3==1,:);
zpv(ind3==1) = 1/(1+w^2) * ( 1 + w*zv(ind3==1,:)./xL2(ind3==1) )*w.*xL2(ind3==1);

% Return results in their original form
% xpv = gather(xpv);
% zpv = gather(zpv);
xpc1 = cat( 4 , reshape( xpv(:,1) , [rows, cols] ), reshape( xpv(:,2) , [rows, cols] ) );
xpc2 = cat( 4 , reshape( xpv(:,3) , [rows, cols] ), reshape( xpv(:,4) , [rows, cols] ) );
xp = cat(3, xpc1, xpc2);
zp = reshape( zpv , size(z) );