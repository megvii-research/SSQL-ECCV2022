from utils.ssl_algorithms import *

def build_ssl_model(base_model, config):
    # create model
    print('creating SSL model {}'.format(config.SSL.TYPE))
    model = eval(config.SSL.TYPE)(base_model, config)
    return model