function W = weight_matrix(x)

eps = 0.001;

[M,N] = size(x);

W = zeros(M,N);

for i = 1:M
    for j = 1:N
        W(i,j) = 1 / sqrt(x(i,j)*x(i,j) + eps*eps);
    end
end
