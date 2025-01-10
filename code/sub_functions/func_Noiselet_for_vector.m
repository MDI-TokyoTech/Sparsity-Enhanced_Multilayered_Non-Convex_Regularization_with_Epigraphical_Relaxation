% input 1: Noiselet matrix for a vector
% input 2: image (matrix)
% Noiselet matrix can be transposed

function[x] = func_Noiselet_for_vector(Noiselet, x)
[rows, cols, dim] = size(x);

for i=1:dim
    xvec = reshape(x(:,:,i), [rows*cols, 1]);
    x(:,:,i) =  reshape(Noiselet*xvec, [rows, cols]);
end