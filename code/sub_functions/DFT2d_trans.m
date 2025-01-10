function Y = DFT2d_trans(X)
% input: 3D matrix (1: real part, 2: imaginary part)
% output: 2D matrix

[M, N, ~] = size(X);

CX = X(:,:,1); % real
SX = X(:,:,2); % imag

% real(Wt) and Wc' is same ( |real(Wt) - Wc'| < 1e-14)
% Wt = conj(dftmtx(M)) / sqrt(M);
% WXWC = real(Wt)*CX;
% WXWS = -imag(Wt)*SX;
Wc = 1/sqrt(M)*cos(2*pi/M*((0:M-1)')*(0:M-1));
Ws = 1/sqrt(M)*sin(2*pi/M*((0:M-1)')*(0:M-1));
WXWC = Wc'*CX;
WXWS = -Ws'*SX;

% transposed DFT for 2D matrix (Z -> X)
% X = W'ZW = (W'(W'Z)')'
Wc2 = 1/sqrt(N)*cos(2*pi/N*((0:N-1)')*(0:N-1));
Ws2 = 1/sqrt(N)*sin(2*pi/N*((0:N-1)')*(0:N-1));
WXWC = Wc'*CX*Wc2 - Ws'*CX*Ws2;
WXWS = -(Wc'*SX*Ws2 + Ws'*SX*Wc2);

% % dimention of X is not suitable
% Wt = dftmtx(M)' / M;
% WXWC = (kron(real(Wt), real(Wt)) - kron(imag(Wt), imag(Wt)))*CX;
% WXWS = -(kron(real(Wt), imag(Wt)) + kron(imag(Wt), real(Wt)))*SX;

Y = WXWC + WXWS;
end

