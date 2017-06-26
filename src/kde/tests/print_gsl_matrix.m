function [] = print_gsl_matrix( matrix, matrix_name )
    for i = 1:size(matrix, 1)
        for j = 1:size(matrix, 2)
            fprintf('\tgsl_matrix_set(%s, %d, %d, %+1.2f);',...
                matrix_name, ...
                i - 1, j - 1, ...
                matrix(i, j)...
            );
        end
        fprintf('\n');
    end
end
