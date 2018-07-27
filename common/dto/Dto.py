class Dto:
    """ Data Transfer Object.
    Usually not required here, but makes it easier for
    passing arguments and consistent naming of variables.
    Allows to iter through its members.
    """
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __repr__(self, indent=None):
        """
        Indicates the fill level, i.e. which attributes
        have non-None values
        :param indent: indent when print out the result
        :return: str representation of the fill level
        """
        result = ''
        if indent is None:
            result += 'Fill level of ' + super().__repr__() + ':\n'
            indent = ''
        for key in sorted(self.__dict__.keys()):
            txt = '[ ]'
            val = self.__dict__[key]
            if val is not None:
                txt = '[x]'
            result += indent + txt + ' ' + key + '\n'
            if isinstance(val, Dto):
                result += val.__repr__(indent=(indent + '    '))
        return result
