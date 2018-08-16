class Dto():
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

    def __str__(self, indent=None):
        """
        Indicates the fill level, i.e. which attributes
        have non-None values
        :param indent: indent when print out the result
        :return: str representation of the fill level
        """
        result = ''
        if indent is None:
            result += 'Fill level of ' + super().__str__() + ':\n'
            indent = ''
        for key in sorted(self.__dict__.keys()):
            txt = '[ ]'
            val = self.__dict__[key]
            if val is not None:
                txt = '[x]'
            result += indent + txt + ' ' + key + '\n'
            if isinstance(val, Dto):
                result += val.__str__(indent=(indent + '    '))
        return result


    def _is_empty(self):
        for key in sorted(self.__dict__.keys()):
            val = self.__dict__[key]
            if val is not None:
                if isinstance(val, Dto):
                    val._is_empty()
                else:
                    return False
        return True
