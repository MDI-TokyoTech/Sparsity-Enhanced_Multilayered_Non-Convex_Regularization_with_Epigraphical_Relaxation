function [ xp , zp ] = Proj_Epigraph_L2_for_ASNN(x, z, w)

[rows, cols, dim] = size(x);

xv = reshape(x, [rows*cols, dim]);
zv = z(:);

xpv = zeros( size(xv) );
zpv = zeros( size(zv) );

xvL2 = sqrt(sum(xv.^2, 2));
ind1 = ( xvL2 < -w*(zv+eps) );
ind2 = ( xvL2 <= (zv+eps)/w );
ind3 = ~or( ind1 , ind2 );

xpv(ind1==1,:) = 0;
zpv(ind1==1) = 0;

xpv(ind2==1,:) = xv(ind2==1,:);
zpv(ind2==1) = zv(ind2==1);

xpv(ind3==1,:) = 1/(1+w^2) * ( 1 + w*zv(ind3==1,:)./sqrt(sum(xv(ind3==1,:).^2,2)) ).*xv(ind3==1,:);
zpv(ind3==1) = 1/(1+w^2) * ( 1 + w*zv(ind3==1,:)./sqrt(sum(xv(ind3==1,:).^2,2)) )*w.*sqrt(sum(xv(ind3==1,:).^2,2));

xp = reshape( xpv , [rows, cols, dim] );
zp = reshape( zpv , size(z) );