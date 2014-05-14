function FI = warped_img(I,u,v)

[M,N] = size(I);

% Calculate frame2to1
[xPosv, yPosv] = meshgrid(1:N,1:M);
xPosv = xPosv - u;
yPosv = yPosv - v;

xPosv(xPosv <= 1) = 1;
yPosv(yPosv <= 1) = 1;
xPosv(xPosv >= N) = N;
yPosv(yPosv >= M) = M;

FI = interp2(I,xPosv,yPosv,'cubic');