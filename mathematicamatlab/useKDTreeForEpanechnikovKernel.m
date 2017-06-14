close all; clear variables; clc;

N = 60000;
d = 3;
p = 10; % 1/p * N points are covered by the kernel

naive = @(N, d, p) N * (N * 2 * d + 1 ./ p * 5 * N);
kdTreeBuild = @(N) N * log(N);
kdTreeLookUp = @(N, d) d * N.^(1 - 1 ./ d);
kdTreeSingle = @(N, d, p) kdTreeLookUp(N, d) + 1 ./ p * N * (2*d + 5);
kdTree = @(N, d, p) kdTreeBuild(N) + N * kdTreeSingle(N, d, p);

pRange = 1:1:10;

hold on

plot(pRange, naive(N, d, pRange));
plot(pRange, kdTree(N, d, pRange));

xlabel('P (fraction of xis covered by the data)');
ylabel('Time complexity');

legend('naive', 'kdtree');

hold off