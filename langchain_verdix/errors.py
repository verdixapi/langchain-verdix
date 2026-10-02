class VerdixError(Exception):
    """A request Verdix refused to make or pay for, or an answer it could not use.

    Raised for bad input, a price above your cap, an exhausted budget,
    unexpected payment terms, or an unexpected API answer. Nothing is paid
    when this is raised before the request is sent.
    """
