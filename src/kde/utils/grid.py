import numpy as np


class Grid(object):
    """Class to represent a uniform grid."""

    def __init__(self, grid_points):
        self._grid_points = grid_points

    @property
    def grid_points(self):
        return self._grid_points

    @classmethod
    def cover(cls, points, padding=0, **kwargs):
        """
        Generate a grid that covers *points* possibly with *padding*.
        :param points: The points that should be covered by the grid.
        :param padding: How much space there is to be between the extrema of points and the extrema of the grid.
        :param kwargs: The parameters that should be passed to the constructor of grid:
            *number_of_grid_points* to indicate that the grid should have a number of vertices or
            *cell_size* to indicate the size of the cells of the grid.
        :return:
        """
        (_, dimensions) = points.shape
        minima = np.min(points, axis=0) - padding
        maxima = np.max(points, axis=0) + padding
        ranges = zip(minima, maxima)
        grid_points = _GridBuilder(ranges, **kwargs).build()
        return cls(grid_points)


class _GridBuilder(object):
    def __init__(self, ranges, **kwargs):
        number_of_grid_points = kwargs.get('number_of_grid_points')
        if(number_of_grid_points):
            self._prepare_with_num_grid_points(number_of_grid_points, ranges)
        else:
            cellsize = kwargs.get('cell_size', 1.0)
            self._prepare_with_cell_size(cellsize, ranges)

    def _prepare_with_num_grid_points(self, number_of_grid_points, ranges):
        self.ranges = list(ranges)
        if np.isscalar(number_of_grid_points):
            self.num_grid_point_list = [number_of_grid_points] * self.grid_dimension
        elif len(number_of_grid_points) is self.grid_dimension:
            self.num_grid_point_list = number_of_grid_points
        else:
            raise ValueError("Number of grid points should be a scalar or a array like with length equal to ranges.")

    def _prepare_with_cell_size(self, cellsize, ranges):
        def compute_grid_domain(minimum, maximum):
            difference = maximum - minimum
            padding = 0.5 * (cellsize - (difference % cellsize))
            return (minimum - padding, maximum + padding)

        def compute_number_of_grid_points(minimum, maximum):
            number_of_grid_points = ((maximum - minimum) / cellsize) + 1
            return number_of_grid_points

        grid_ranges = list()
        number_of_grid_points = list()
        for (minimum, maximum) in ranges:
            grid_range = compute_grid_domain(minimum, maximum)
            grid_ranges.append(grid_range)
            number_of_grid_points.append(compute_number_of_grid_points(grid_range[0], grid_range[1]))
        self._prepare_with_num_grid_points(number_of_grid_points, grid_ranges)

    @property
    def grid_dimension(self):
        return len(self.ranges)

    def build(self):
        points = list()

        # Generate the points for all dimensions
        for ((minimum, maximum), num_grid_points) in zip(self.ranges, self.num_grid_point_list):
            points.append(np.linspace(start=minimum, stop=maximum, num=num_grid_points))

        # Generate the list of grid positions
        grid = np.meshgrid(*points, indexing='xy')
        positions = np.vstack(map(np.ravel, grid)).transpose()
        return positions
