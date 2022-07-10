class MeterBasic(object):
    """Meters provide a way to keep track of important statistics in an online manner.

    This class is abstract, but provides a standard interface for all meters to follow.

    """

    def reset(self):
        """Resets the meter to default settings."""
        pass

    def update(self, value, *args):
        """Log a new value to the meter

        Args:
            value: Next restult to include.

        """
        pass

    def value(self):
        """Get the value of the meter in the current state."""
        pass

    def __repr__(self):
        """Get/Print the infomation of the meter in the current state."""
        return ""

    def update_best_meter_dict(self):
        self.is_best = False

    def get_best_values(self):
        return self.best_meter_dict
