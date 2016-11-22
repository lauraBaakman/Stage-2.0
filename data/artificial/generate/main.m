generators = {@ferdosi_1, @ferdosi_2, @ferdosi_3, @ferdosi_4, @ferdosi_5};
n = 60000;
d = 3;
basename = '../';

 for i = 1:length(generators)
    [data, densities, ns] = generators{i}(n, d);
    write_to_file(data, densities, ns, sprintf('%s%s_%d.txt', basename, func2str(generators{i}), n))
end