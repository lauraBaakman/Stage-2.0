import numpy as np


class Grid(object):
    """Class to represent a uniform grid."""

    def __init__(self, number_of_grid_points, *ranges):
        """
        Init method of Grid.
        :param number_of_grid_points: How many grid points should be used to cover one dimension, can be either a single
        number or a list of N values. Note that the key word is required for this argument.
        :param ranges: List of N tuples, each representing the start and the end of a range. A N-dimensional grid
        is generated.
        """
        self._ranges = ranges
        self._number_of_grid_points = number_of_grid_points
        self._grid_points = _GridBuilder(self._number_of_grid_points, self._ranges).build()

    @property
    def grid_points(self):
        return self._grid_points

    @classmethod
    def cover(cls, points, padding=0, **kwargs):
        """
        Generate a grid that covers *points* possibly with *padding*.
        :param points: The points that should be covered by the grid.
        :param padding: How much space there is to be between the extrema of points and the extrema of the grid.
        :param kwargs: The parameters that should be passed to the constructor of grid: *number_of_grid_points*
        :return:
        """
        (_, dimensions) = points.shape
        minima = np.min(points, axis=0) - padding
        maxima = np.max(points, axis=0) + padding
        ranges = zip(minima, maxima)
        return cls(kwargs['number_of_grid_points'], *ranges)


class _GridBuilder(object):
    def __init__(self, number_of_grid_points, *ranges):
        self.ranges = list(ranges[0])

        if np.isscalar(number_of_grid_points):
            self.num_grid_point_list = [number_of_grid_points] * self.grid_dimension
        elif len(number_of_grid_points) is self.grid_dimension:
            self.num_grid_point_list = number_of_grid_points
        else:
            raise ValueError("Number of grid points should be a scalar or a array like with length equal to ranges.")

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
