import abc

from paddle import nn


class InterpolationBase(nn.Layer, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def grid_points(self):
        """The time points.

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def interval(self):
        """The time interval between time points.

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def interpolate(self, t):
        """Calculates the index of the given time point t in the list of time points.

        Args:
            t (_type_): time point t

        Raises:
            NotImplementedError:

        Retuns:
            The index of the given time point t in the list of time points.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, t):
        """Calculates the value at the time point t.

        Args:
            t (_type_): The time point t

        Raises:
            NotImplementedError:

        Retruns:
            The value at the time point t.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def derivative(self, t):
        """Calculates the derivative of the function at the point t.

        Args:
            t (_type_): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            The derivative of the function at the point t.
        """
        raise NotImplementedError
