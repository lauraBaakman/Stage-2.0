close all; clear variables; clc;

kernel = @(x) mvnpdf(x);

mbe_singlePattern = @(x, xs, h, l, kernel) (l * h)^(-length(x)) * kernel((x - xs) ./ (l * h));

mbe = @(x, xs, general_bandwidth, local_bandwidths, kernel) 1 / 3 * (...
    mbe_singlePattern(x, xs(1, :), general_bandwidth, local_bandwidths(1), kernel) + ...
    mbe_singlePattern(x, xs(2, :), general_bandwidth, local_bandwidths(2), kernel) + ...
    mbe_singlePattern(x, xs(3, :), general_bandwidth, local_bandwidths(3), kernel));

xi_s = [-1, -1; ...
        +1, +1; ...
        +0, +0];
x_s = [0, 0; ...
       1, 1; ...
       0, 1];
local_bandwidths = [10, 20, 50];
general_bandwidth = 0.5;


print_gsl_matrix(x_s, 'xs');
fprintf('\n')

print_gsl_matrix(xi_s, 'xis');

fprintf('\n');
for i = 1:size(x_s, 1)
    fprintf('gsl_vector_set(expected, %d, %2.15f);\n', ...
        i - 1, ...
        mbe(x_s(i, :), xi_s, general_bandwidth, local_bandwidths, kernel));
end