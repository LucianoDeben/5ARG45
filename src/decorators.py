import logging
import time


def debug(func):
    def wrapper(*args, **kwargs):
        logging.info(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        logging.info(f"Finished {func.__name__}")
        return result

    return wrapper


def errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            raise

    return wrapper


def shapes(func):
    def wrapper(*args, **kwargs):
        logging.info(
            f"Input shapes: {[arg.shape for arg in args if hasattr(arg, 'shape')]}"
        )
        result = func(*args, **kwargs)
        if hasattr(result, "shape"):
            logging.info(f"Output shape: {result.shape}")
        return result

    return wrapper


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result

    return wrapper


import torch


def gpu_memory(func):
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            logging.info(
                f"GPU memory allocated before: {torch.cuda.memory_allocated()}"
            )
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            logging.info(f"GPU memory allocated after: {torch.cuda.memory_allocated()}")
        return result

    return wrapper
