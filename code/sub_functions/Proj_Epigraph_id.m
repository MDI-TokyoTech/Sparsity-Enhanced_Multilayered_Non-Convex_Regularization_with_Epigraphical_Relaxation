function [ xp , zp ] = Proj_Epigraph_id( x , z )

xv = reshape( x, [numel(x),1] );
zv = reshape( z ,[numel(z),1] );

% set horizontal vector as 2 rows
xz = [ xv' ; zv' ];

% extract the points being out of the epigraph (xv != zv)
ind = ( xv > zv );
xz_ = xz(:,ind);

% xzp_ = xz_ - a *([ 1/2 -1/2 ]a'*xz_);
% a = [ 1 ; -1 ];
xzp_ = xz_ - [ 1 ; -1 ]*([ 1/2 -1/2 ]*xz_);
xzp_(2,:) = xzp_(1,:); % eliminate a little error caused by Computer

% assign results of epigraph projection
xzp = xz; 
xzp(:,ind) = xzp_;

% extract x,z as vertical vector
xpv = xzp(1,:); xpv = xpv';
zpv = xzp(2,:); zpv = zpv';

% reshape to original form
xp = reshape( xpv , size(x) );
zp = reshape( zpv , size(z) );
