import signal
import functools
import os

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message='Function call timed out'):
    def decorator(func):
        if os.name == 'nt':  # Windows does not support SIGALRM
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(error_message)

            # Set the signal handler and alarm
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Disable the alarm
                signal.alarm(0)
            return result
        return wrapper
    return decorator
