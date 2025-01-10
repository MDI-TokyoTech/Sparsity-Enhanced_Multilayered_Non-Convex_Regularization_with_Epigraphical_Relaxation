% Only for 2^n square image

function[x] = func_NoiseletMatrix(Noiselet, x)
for i=1:size(x,3)
    x(:,:,i) = Noiselet*x(:,:,i);
end