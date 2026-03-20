import numpy as np

from pymo.Quaternions import Quaternions

class Pivots:
    """
    Pivots is an ndarray of angular rotations

    This wrapper provides some functions for
    working with pivots.

    These are particularly useful as a number
    of atomic operations (such as adding or
    subtracting) cannot be achieved using
    the standard arithmatic and need to be
    defined differently to work correctly
    """

    def __init__(self, ps):
        """
        Initialize rotation axis class.

        Args:
        ps: Array of rotation axis angles.
        """
        self.ps = np.array(ps)

    def __str__(self):
        """
        Represent rotation axis as string.

        Returns:
        String representation of the rotation axis angles array.
        """
        return "Pivots(" + str(self.ps) + ")"

    def __repr__(self):
        """
        Represent rotation axis as expression.

        Returns:
        Expression string of the rotation axis angles array.
        """
        return "Pivots(" + repr(self.ps) + ")"

    def __add__(self, other):
        """
        Implement addition of two rotation axes.

        Args:
        other: Another rotation axis.

        Returns:
        New rotation axis object after addition.
        """
        return Pivots(np.arctan2(np.sin(self.ps + other.ps), np.cos(self.ps + other.ps)))

    def __sub__(self, other):
        """
        Implement subtraction of two rotation axes.

        Args:
        other: Another rotation axis.

        Returns:
        New rotation axis object after subtraction.
        """
        return Pivots(np.arctan2(np.sin(self.ps - other.ps), np.cos(self.ps - other.ps)))

    def __mul__(self, other):
        """
        Implement multiplication of two rotation axes.

        Args:
        other: Another rotation axis.

        Returns:
        New rotation axis object after multiplication.
        """
        return Pivots(self.ps * other.ps)

    def __div__(self, other):
        """
        Implement division of two rotation axes.

        Args:
        other: Another rotation axis.

        Returns:
        New rotation axis object after division.
        """
        return Pivots(self.ps / other.ps)

    def __mod__(self, other):
        """
        Implement modulo operation of two rotation axes.

        Args:
        other: Another rotation axis.

        Returns:
        New rotation axis object after modulo.
        """
        return Pivots(self.ps % other.ps)

    def __pow__(self, other):
        """
        Implement power operation of two rotation axes.

        Args:
        other: Another rotation axis.

        Returns:
        New rotation axis object after power operation.
        """
        return Pivots(self.ps ** other.ps)

    def __lt__(self, other):
        """
        Compare two rotation axes by angle (less than).

        Args:
        other: Another rotation axis.

        Returns:
        Boolean array of comparison results.
        """
        return self.ps < other.ps

    def __le__(self, other):
        """
        Compare two rotation axes by angle (less than or equal).

        Args:
        other: Another rotation axis.

        Returns:
        Boolean array of comparison results.
        """
        return self.ps <= other.ps

    def __eq__(self, other):
        """
        Compare two rotation axes by angle (equal).

        Args:
        other: Another rotation axis.

        Returns:
        Boolean array of comparison results.
        """
        return self.ps == other.ps

    def __ne__(self, other):
        """
        Compare two rotation axes by angle (not equal).

        Args:
        other: Another rotation axis.

        Returns:
        Boolean array of comparison results.
        """
        return self.ps != other.ps

    def __ge__(self, other):
        """
        Compare two rotation axes by angle (greater than or equal).

        Args:
        other: Another rotation axis.

        Returns:
        Boolean array of comparison results.
        """
        return self.ps >= other.ps

    def __gt__(self, other):
        """
        Compare two rotation axes by angle (greater than).

        Args:
        other: Another rotation axis.

        Returns:
        Boolean array of comparison results.
        """
        return self.ps > other.ps

    def __abs__(self):
        """
        Calculate absolute value of each rotation axis angle.

        Returns:
        Rotation axis object with absolute values.
        """
        return Pivots(abs(self.ps))

    def __neg__(self):
        """
        Negate each rotation axis angle.

        Returns:
        Rotation axis object with negated values.
        """
        return Pivots(-self.ps)

    def __iter__(self):
        """
        Implement iteration over rotation axis.

        Returns:
        Iterator over the rotation axis angles array.
        """
        return iter(self.ps)

    def __len__(self):
        """
        Get length of rotation axis angles array.

        Returns:
        Length of the rotation axis angles array.
        """
        return len(self.ps)

    def __getitem__(self, k):
        """
        Get rotation axis angle by index.

        Args:
        k: Index or slice.

        Returns:
        Rotation axis object at the specified index.
        """
        return Pivots(self.ps[k])

    def __setitem__(self, k, v):
        """
        Set rotation axis angle by index.

        Args:
        k: Index or slice.
        v: Rotation axis object to set.
        """
        self.ps[k] = v.ps

    def _ellipsis(self):
        """
        Get ellipsis slice object for indexing.

        Returns:
        Tuple of ellipsis slice objects.
        """
        return tuple(map(lambda x: slice(None), self.ps.shape))

    def quaternions(self, plane='xz'):
        """
        Convert rotation axis to quaternions.

        Args:
        plane: Specified rotation plane, default 'xz'.

        Returns:
        Quaternion object converted from rotation axis.
        """
        fa = self._ellipsis()
        axises = np.ones(self.ps.shape + (3,))
        axises[fa + ("xyz".index(plane[0]),)] = 0.0
        axises[fa + ("xyz".index(plane[1]),)] = 0.0
        return Quaternions.from_angle_axis(self.ps, axises)

    def directions(self, plane='xz'):
        """
        Get direction vectors of rotation axis in specified plane.

        Args:
        plane: Specified rotation plane, default 'xz'.

        Returns:
        Direction vectors array of rotation axis in specified plane.
        """
        dirs = np.zeros((len(self.ps), 3))
        dirs["xyz".index(plane[0])] = np.sin(self.ps)
        dirs["xyz".index(plane[1])] = np.cos(self.ps)
        return dirs

    def normalized(self):
        """
        Normalize rotation axis angles.

        Returns:
        Normalized rotation axis object.
        """
        xs = np.copy(self.ps)
        while np.any(xs > np.pi): xs[xs > np.pi] -= 2 * np.pi
        while np.any(xs < -np.pi): xs[xs < -np.pi] += 2 * np.pi
        return Pivots(xs)

    def interpolate(self, ws):
        """
        Linearly interpolate rotation axis.

        Args:
        ws: Weight array for interpolation calculation.

        Returns:
        Interpolated rotation axis angle.
        """
        dir = np.average(self.directions, weights=ws, axis=0)
        return np.arctan2(dir[2], dir[0])

    def copy(self):
        """
        Copy rotation axis object.

        Returns:
        Copied rotation axis object.
        """
        return Pivots(np.copy(self.ps))

    @property
    def shape(self):
        """
        Get shape of rotation axis angles array.

        Returns:
        Shape of the rotation axis angles array.
        """
        return self.ps.shape

    @classmethod
    def from_quaternions(cls, qs, forward='z', plane='xz'):
        """
        Create rotation axis from quaternions.

        Args:
        qs: Quaternion array.
        forward: Specified forward direction, default 'z'.
        plane: Specified rotation plane, default 'xz'.

        Returns:
        Rotation axis object converted from quaternions.
        """
        ds = np.zeros(qs.shape + (3,))
        ds[...,'xyz'.index(forward)] = 1.0
        return Pivots.from_directions(qs * ds, plane=plane)

    @classmethod
    def from_directions(cls, ds, plane='xz'):
        """
        Create rotation axis from direction vectors.

        Args:
        ds: Direction vectors array.
        plane: Specified rotation plane, default 'xz'.

        Returns:
        Rotation axis object converted from direction vectors.
        """
        ys = ds[...,'xyz'.index(plane[0])]
        xs = ds[...,'xyz'.index(plane[1])]
        return Pivots(np.arctan2(ys, xs))
