import sys


TAB = ' ' * 2


class Parameter:
    def __init__(self,
                 name,
                 key,
                 type,
                 default=None,
                 dependencies=None,
                 help_message=None):
        self.name = name
        self.key = key
        self.type = type
        self.default = default
        self.dependencies = dependencies
        self.help_message = help_message

    def _get_from_args(self, args):
        value = args[self.key]
        return _cast(value, self.type)

    def _get_from_console(self):
        if self.help_message is not None:
            print(self.help_message)
        if self.default is None:
            msg = "%s: " % self.name
        else:
            msg = "%s (default: %s): " % (self.name, str(self.default))
        value = input(msg)
        if value == "":
            return self.default
        elif ' ' in value:
            value = value.split(' ')
        return _cast(value, self.type)

    def register(self, settings, args):
        if self.key in args:
            settings[self.name] = self._get_from_args(args)
        else:
            settings[self.name] = self._get_from_console()
        self._register_dependencies(settings, args)

    def _register_dependencies(self, settings, args):
        if self.dependencies is not None:
            for dep_val, params in self.dependencies:
                if not isinstance(params, list):
                    params = [params]
                for param in params:
                    if settings[self.name] == dep_val:
                        param.register(settings, args)

    def print_help(self, dependecy_depth=0):
        spacing = TAB * dependecy_depth * 2
        subspacing = spacing + TAB
        print(spacing + self.key)
        print(subspacing + "name: " + self.name)
        print(subspacing + "type: " + self.type)
        if self.default is not None:
            print(subspacing + "default: " + str(self.default))
        if self.help_message is not None:
            print(subspacing + self.help_message)
        if self.dependencies is not None:
            for dep_val, params in self.dependencies:
                print(subspacing + "if %s:" % str(dep_val))
                if not isinstance(params, list):
                    params = [params]
                for param in params:
                    param.print_help(dependecy_depth=dependecy_depth+1)


def _is_key(arg):
    return len(arg) > 1 and arg[0] == "-" and not arg[1].isnumeric()


def _parse_args():
    args = {}
    key = None
    for arg in sys.argv[1:]:
        if _is_key(arg):
            key = arg
        else:
            if key not in args:
                args[key] = arg
            elif not isinstance(args[key], list):
                args[key] = [args[key], arg]
            else:
                args[key].append(arg)
    return args


def _cast_single_bool(value):
    if value in ["1", "True"]:
        return True
    if value in ["0", "False"]:
        return False
    raise Exception("Unable to cast '%s' to bool." % str(value))


def _cast_single(value, type):
    if type == "str":
        return value
    if type == "int":
        return int(value)
    if type == "float":
        return float(value)
    if type == "boolean":
        return _cast_single_bool(value)
    raise NotImplementedError("Unknown parameter type %s." % str(type))


def _cast(value, type):
    if isinstance(value, list):
        return [_cast_single(elem, type) for elem in value]
    else:
        return _cast_single(value, type)


class ParameterGetter:
    def __init__(self, parameters):
        self.parameters = parameters
        if "-h" in sys.argv:
            self.print_help()
            exit(1)

    def get(self):
        settings = {}
        args = _parse_args()
        for param in self.parameters:
            param.register(settings, args)
        return settings

    def print_help(self):
        print("Parameters:")
        for param in self.parameters:
            param.print_help()


if __name__ == "__main__":
    parser = ParameterGetter([Parameter("model_name", "-name", "str",
                                        help_message="The name of the model."),
                              Parameter("model_type", "-type", "str",
                                        dependencies=[("attention", [Parameter("attention_type", "-at", "str"),
                                                                     Parameter("heads", "-h", "int", default=1)])],
                                        help_message="The model's type. Choose between 'lstm' and 'attention'."),
                              Parameter("hidden_units", "-units", "int", [0, 1, 2],
                                        help_message="Number of units per layer.")])
    parser.print_help()
    print(parser.get())
