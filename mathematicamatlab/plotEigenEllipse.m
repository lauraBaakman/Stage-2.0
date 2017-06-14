clear variables; close all; clc;

xs = normrnd(0, 5, 1000, 1);
ys = normrnd(0, 2, 1000, 1);

covarianceMatrix = cov([xs, ys]);

%% Draw the eigen ellipse of the covariance matrix
[eigenVectors, eigenValues] = eig(covarianceMatrix);
eigenValues = diag(eigenValues);

phi = atan2(eigenVectors(2,end), eigenVectors(1,end));
if(phi < 0)
    phi = phi + 2*pi;
end

theta_grid = linspace(0,2*pi, 200);
a=sqrt(eigenValues(2));
b=sqrt(eigenValues(1));

ellipse_x_r  = a * cos(theta_grid);
ellipse_y_r  = b * sin(theta_grid);


rotation = [ cos(phi) sin(phi); -sin(phi) cos(phi) ];
r_ellipse = [ellipse_x_r; ellipse_y_r]' * rotation;

hold on
axis equal
plot3(r_ellipse(:,1), r_ellipse(:,2), ones(size(r_ellipse(:,1))));


%% Draw the isolines of the Gaussian Kernel as defined by H?rdle, Muller et al.

[x, y] = meshgrid(linspace(min(xs), max(xs), 400),linspace(min(ys), max(ys), 400));
locations = [x(:) y(:)];

z = mvnpdf(locations * inv(covarianceMatrix));
z = reshape(z,size(x));
contour(x, y, z)

%% Draw the data points
plot3(xs, ys, ones(length(xs)), '*');

hold off
