function out = adj_shift(vid1,vid2,M,N)

n = length(vid1);
out = [];

for i = 1:n
    out{i} = zeros(M,N,3);
    [ssde_r,out{i}(:,:,1)] = comp_upto_shift(vid1{i}(:,:,1),vid2{i}(:,:,1));
    [ssde_g,out{i}(:,:,2)] = comp_upto_shift(vid1{i}(:,:,2),vid2{i}(:,:,2));
    [ssde_b,out{i}(:,:,3)] = comp_upto_shift(vid1{i}(:,:,3),vid2{i}(:,:,3));
end