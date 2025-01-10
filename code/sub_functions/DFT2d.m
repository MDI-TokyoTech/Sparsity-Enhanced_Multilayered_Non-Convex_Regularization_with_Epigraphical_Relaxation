function y = DFT2d(X)
% input: 2D matrix
% output: 3D matrix (1: real part, 2: imaginary part)

[M, N] = size(X);

% real(W) and Wc is same ( |real(W) - Wc| < 1e-14)
% W = dftmtx(M) / sqrt(M);
% WXWC = real(W)*X;
% WXWS = -imag(W)*X;
Wc = 1/sqrt(M)*cos(2*pi/M*((0:M-1)')*(0:M-1));
Ws = 1/sqrt(M)*sin(2*pi/M*((0:M-1)')*(0:M-1));
WXWC = Wc*X;
WXWS = -Ws*X;

% DFT for 2D matrix (X -> Z)
% Z = WXW' = (W(WX)')'
Wc2 = 1/sqrt(N)*cos(2*pi/N*((0:N-1)')*(0:N-1));
Ws2 = 1/sqrt(N)*sin(2*pi/N*((0:N-1)')*(0:N-1));
WXWC = Wc*X*Wc2' - Ws*X*Ws2';
WXWS = -(Wc*X*Ws2' + Ws*X*Wc2');

% % dimention of X is not suitable
% W = dftmtx(M) / sqrt(M);
% WXWC = (kron(real(W), real(W)) - kron(imag(W), imag(W)))*X;
% WXWS = -(kron(real(W), imag(W)) + kron(imag(W), real(W)))*X;

% concat for Proj_epigraph_L2
y = cat(3, WXWC, WXWS);
end

