from typing import Any, Callable, List


def ensure_list(input: Any) -> List:
    """
    Ensures that the input is wrapped in a list. If the input is None, returns an empty list.

    Args:
        input: The input to wrap in a list.

    Returns:
        List: The input wrapped in a list, or an empty list if input is None.
    """
    if isinstance(input, list):
        return input
    if isinstance(input, tuple):
        return list(input)
    if input is None:
        return []
    return [input]


def apply_fns(data: Any, fns: Callable | List[Callable]) -> Any:
    """
    Applies a function or a list of functions to the input data.

    Args:
        data: The data to process.
        fns: A function or list of functions to apply.

    Returns:
        Any: The processed data after applying the function(s).
    """
    for fn in ensure_list(fns):
        data = fn(data)
    return data
