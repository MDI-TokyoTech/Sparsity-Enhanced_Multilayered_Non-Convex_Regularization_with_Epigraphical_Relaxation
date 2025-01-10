function [ x , y ] = ProjHalfspace( x, y , mu )

X = cat( 3 , x , y );

if sum(X(:)) > mu
    X = X + (mu-sum(X(:)))/numel(X)*ones(size(X));
end

x = X(:,:,1);
y = X(:,:,2);