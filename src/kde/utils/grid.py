import numpy as np

_default_number_of_grid_points = 20


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
        cell_size = kwargs.get('cell_size')
        if(cell_size):
            grid_ranges = self._compute_grid_ranges(cell_size, ranges)
            number_of_grid_points = self._compute_number_of_grid_points(cell_size, grid_ranges)
        else:
            grid_ranges = ranges
            number_of_grid_points = kwargs.get('number_of_grid_points', _default_number_of_grid_points)
        self.num_grid_point_list = self._compute_num_grid_point_list(number_of_grid_points, grid_ranges)

    def _compute_num_grid_point_list(self, number_of_grid_points, ranges):
        self.ranges = list(ranges)
        if np.isscalar(number_of_grid_points):
            return [number_of_grid_points] * self.grid_dimension
        elif len(number_of_grid_points) is self.grid_dimension:
            return number_of_grid_points
        else:
            raise ValueError("Number of grid points should be a scalar or a array like with length equal to ranges.")

    def _compute_grid_ranges(self, cellsize, ranges):
        def compute_grid_range(minimum, maximum):
            difference = maximum - minimum
            padding = 0.5 * (cellsize - (difference % cellsize))
            return (minimum - padding, maximum + padding)

        grid_ranges = list()
        for (minimum, maximum) in ranges:
            grid_ranges.append(compute_grid_range(minimum, maximum))
        return grid_ranges

    def _compute_number_of_grid_points(self, cellsize, grid_ranges):
        def compute_number_of_grid_points_for_range(minimum, maximum):
            number_of_grid_points = ((maximum - minimum) / cellsize) + 1
            return number_of_grid_points

        number_of_grid_points = list()
        for (minimum, maximum) in grid_ranges:
            number_of_grid_points.append(compute_number_of_grid_points_for_range(minimum, maximum))
        return number_of_grid_points

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
