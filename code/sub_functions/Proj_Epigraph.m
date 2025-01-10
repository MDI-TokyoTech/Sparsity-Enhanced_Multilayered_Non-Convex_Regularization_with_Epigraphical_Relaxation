
function [ xp , zp ] = Proj_Epigraph(x, z, w)

if ~exist('w','var')
    w = 1;
end

[ Ny , Nx , Nz ] = size(x);

if Nz > 1
    xv = reshape( x, [Ny*Nx,Nz] );
    zv = reshape( z ,[Ny*Nx,1] );
else 
    xv = x;
    zv = z;
end

xpv = zeros( size(xv) );
zpv = zeros( size(zv) );

ind1 = ( sqrt(sum(xv.^2,2)) < -w*(zv+eps) );
ind2 = ( sqrt(sum(xv.^2,2)) < (zv+eps)/w );
ind3 = ~or( ind1 , ind2 );

xpv(ind1==1,:) = 0;
zpv(ind1==1) = 0;

xpv(ind2==1,:) = xv(ind2==1,:);
zpv(ind2==1) = zv(ind2==1);

xpv(ind3==1,:) = 1/(1+w^2) * ( 1 + w*zv(ind3==1,:)./sqrt(sum(xv(ind3==1,:).^2,2)) ).*xv(ind3==1,:);
zpv(ind3==1) = 1/(1+w^2) * ( 1 + w*zv(ind3==1,:)./sqrt(sum(xv(ind3==1,:).^2,2)) )*w.*sqrt(sum(xv(ind3==1,:).^2,2));

if Nz > 1
    xp = reshape( xpv , size(x) );
    zp = reshape( zpv , size(z) );
%     xp = reshape( xpv , [Ny,Nx,Nz] );
%     zp = reshape( zpv , [Ny,Nx] );
else
    xp = xpv;
    zp = zpv;
end