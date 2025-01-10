function [ xp , zp ] = Proj_Epigraph_L2_for_DSTV_chro(x, z, w)
% L2 projection when x is a 6D vector
% x = (rows, cols, color, vh, kernel_rows, kernel_cols)

if ~exist('w','var')
    w = 1;
end

% Create difference vectors and Set side by side
xv = cat( 2, reshape( x(:,:,1,:,:,:), [ numel(z), 1 ] ), reshape( x(:,:,2,:,:,:), [ numel(z), 1] ) );
zv = reshape( z ,[ numel(z) , 1]  );

xpv = zeros( size(xv) );
zpv = zeros( size(zv) );

% Check each element if it is in the epigraph
ind1 = ( sqrt(sum(xv.^2,2)) < -w*(zv+eps) );
ind2 = ( sqrt(sum(xv.^2,2)) <= (zv+eps)/w );
ind3 = ~or( ind1 , ind2 );

% Cast to 0
xpv(ind1==1,:) = 0;
zpv(ind1==1) = 0;

% If it is in the epigraph, it stays.
xpv(ind2==1,:) = xv(ind2==1,:);
zpv(ind2==1) = zv(ind2==1);

% Cast to nearest point in the epigraph
xpv(ind3==1,:) = 1/(1+w^2) * ( 1 + w*zv(ind3==1,:)./sqrt(sum(xv(ind3==1,:).^2,2)) ).*xv(ind3==1,:);
zpv(ind3==1) = 1/(1+w^2) * ( 1 + w*zv(ind3==1,:)./sqrt(sum(xv(ind3==1,:).^2,2)) )*w.*sqrt(sum(xv(ind3==1,:).^2,2));

% Return results in their original form
xp = cat( 3 , reshape( xpv(:,1) , size(z) ), reshape( xpv(:,2) , size(z) ) );
zp = reshape( zpv , size(z) );