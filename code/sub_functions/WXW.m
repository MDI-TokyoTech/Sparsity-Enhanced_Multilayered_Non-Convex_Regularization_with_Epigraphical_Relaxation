function y = WXW(x,N)
K = size(x,2);
X = reshape(x,[N,N,K]);

WXWC = @(z,N) real(dftmtx(N)')*z*(real(dftmtx(N)'))' - imag(dftmtx(N)')*z*(imag(dftmtx(N)'))';
WXWS = @(z,N) real(dftmtx(N)')*z*(imag(dftmtx(N)'))' - imag(dftmtx(N)')*z*real(dftmtx(N)');

for i = 1:K
    CX(:,:,i) = WXWC(X(:,:,i),N);
    SX(:,:,i) = WXWS(X(:,:,i),N);
end

y = cat(3,reshape(CX,[N*N,K]),reshape(SX,[N*N,K]));