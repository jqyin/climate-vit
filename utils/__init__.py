
def get_data_loader_distributed(params, location, distributed, train):
    from .data_loader import get_data_loader
    return get_data_loader(params, location, distributed, train)

