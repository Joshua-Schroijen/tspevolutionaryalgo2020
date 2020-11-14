import cProfile, pstats

def profile(filename):
    def _profile(function):
        def wrapped(*args, **kwargs):
            profile = cProfile.Profile()
            profile.enable()
            retval = function(*args, **kwargs)
            profile.disable()
            output_file = open(filename, "w", encoding="utf-8")
            stats = pstats.Stats(profile, stream=output_file).sort_stats("cumulative")
            stats.print_stats()
            output_file.close()

            return retval

        return wrapped

    return _profile