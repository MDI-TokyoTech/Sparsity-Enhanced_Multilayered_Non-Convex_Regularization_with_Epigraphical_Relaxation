% Only for 2^n square image

function[x] = func_NoiseletMatrixTrans(Noiselet, x)
for i=1:size(x,3)
    x(:,:,i) = (Noiselet)'*x(:,:,i);
end
x = x/size(x,1);