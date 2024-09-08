def route(rule, methods):
    def decorator(func):
        setattr(func, 'rule', rule)
        setattr(func, 'method_types', methods)
        return func

    return decorator
