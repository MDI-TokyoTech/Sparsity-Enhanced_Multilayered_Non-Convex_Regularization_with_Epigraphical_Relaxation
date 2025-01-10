function [ xp , zp ] = Proj_Epigraph_Liner( x , z, slope, intercept )

if ~exist('slope','var')
    slope = 1;
end
if ~exist('intercept','var')
    intercept = 0;
end

xv = reshape( x, [numel(x),1] );
zv = reshape( z ,[numel(z),1] );
zv = zv - intercept;

% set horizontal vector as 2 rows
xz = [ xv' ; zv' ];

% extract the points being out of the epigraph (xv != zv)
ind = ( slope*xv > zv );
xz_ = xz(:,ind);

a = [slope; -1];
% xzp_ = xz_ - a *([ 1/2 -1/2 ]a'*xz_);
xzp_ = xz_ - a*((a'*xz_)/(a'*a));
xzp_(2,:) = xzp_(1,:); % eliminate a little error caused by Computer

% assign results of epigraph projection
xzp = xz; 
xzp(:,ind) = xzp_;

% extract x,z as vertical vector
xpv = xzp(1,:); xpv = xpv';
zpv = xzp(2,:); zpv = zpv';

% reshape to original form
xp = reshape( xpv , size(x) );
zp = reshape( zpv , size(z) ) + intercept;