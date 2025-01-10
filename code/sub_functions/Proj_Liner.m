function [ xp , zp ] = Proj_Liner( x , z, slope, intercept )

if ~exist('slope','var')
    slope = 1;
end
if ~exist('intercept','var')
    intercept = 0;
end

xv = reshape( x, [numel(x),1] );
zv = reshape( z ,[numel(z),1] );
zv = zv - intercept;

a = [slope; -1];

% set x and z as horizontal vector (2 rows)
xz = [ xv' ; zv' ];

%---------------------------------------------------------
%  Projection
%---------------------------------------------------------
% extract the points that out of the line (xv != zv)
ind = ( slope*xv ~= zv );
xz_ = xz(:,ind);

xzp_ = xz_ - a*((a'*xz_)/(a'*a));
xzp_(2,:) = xzp_(1,:); % eliminate little errors caused by Computer

% assign results of epigraph projection
xzp = xz;
xzp(:,ind) = xzp_;

%---------------------------------------------------------
%  return results
%---------------------------------------------------------
% extract x,z as vertical vector
xpv = xzp(1,:); xpv = xpv';
zpv = xzp(2,:); zpv = zpv';

% reshape to original form
xp = reshape( xpv , size(x) );
zp = reshape( zpv , size(z) ) + intercept;