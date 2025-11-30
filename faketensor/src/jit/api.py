from .base import trace_mode, freeze_graph

# ================================================================
# PUBLIC JIT DECORATOR
# ================================================================
def jit(fun):
    cache = {}  # shape-polymorphic cache: type structure → compiled graph

    def wrapped(*args):
        key = tuple(type(a) for a in args)

        if key not in cache:
            # First run → trace + compile
            with trace_mode():
                out = fun(*args)
                graph = freeze_graph(args, out)

            cache[key] = graph
            return out

        # Fast path
        return cache[key](*args)

    return wrapped
