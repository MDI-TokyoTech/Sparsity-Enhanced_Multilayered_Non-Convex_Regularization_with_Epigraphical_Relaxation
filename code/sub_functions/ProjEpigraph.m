
function [ xp , zp ] = ProjEpigraph(x, z, w)

[ Ny , Nx , Nz ] = size(x);
% xp1 = zeros( size(x) );
% zp1 = zeros( size(z) );
% 
% for ny = 1:Ny
%     for nx = 1:Nx
%         currx = squeeze(x(ny,nx,:));
%         if norm(currx,2) < -w*(z(ny,nx)+eps)
%             xp1(ny,nx,:) = 0;
%             zp1(ny,nx) = 0;
%         elseif norm(currx,2) < (z(ny,nx)+eps)/w
%             xp1(ny,nx,:) = x(ny,nx,:);
%             zp1(ny,nx) = z(ny,nx);
%         else
%             xp1(ny,nx,:) = 1/(1+w^2) * ( 1 + w*z(ny,nx)/(norm(currx,2)) )*x(ny,nx,:);
%             zp1(ny,nx) = 1/(1+w^2) * ( 1 + w*z(ny,nx)/(norm(currx,2)) )*w*norm(currx,2);
%         end
%     end
% end

xv = reshape( x, [Ny*Nx,Nz] );
zv = reshape( z ,[Ny*Nx,1] );

xpv = zeros( size(xv) );
zpv = zeros( size(zv) );

ind1 = ( sqrt(sum(xv.^2,2)) < -w*(zv+eps) );
ind2 = ( sqrt(sum(xv.^2,2)) <= (zv+eps)/w );
ind3 = ~or( ind1 , ind2 );

xpv(ind1==1,:) = 0;
zpv(ind1==1) = 0;

xpv(ind2==1,:) = xv(ind2==1,:);
zpv(ind2==1) = zv(ind2==1);

xpv(ind3==1,:) = 1/(1+w^2) * ( 1 + w*zv(ind3==1,:)./sqrt(sum(xv(ind3==1,:).^2,2)) ).*xv(ind3==1,:);
zpv(ind3==1) = 1/(1+w^2) * ( 1 + w*zv(ind3==1,:)./sqrt(sum(xv(ind3==1,:).^2,2)) )*w.*sqrt(sum(xv(ind3==1,:).^2,2));

xp = reshape( xpv , [Ny,Nx,Nz] );
zp = reshape( zpv , [Ny,Nx] );
