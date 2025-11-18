from torch.func import functional_call


def fast_functional_forward(model, x, params):
    # params ist eine Liste in Reihenfolge von model.parameters()
    # wir müssen sie in ein dict name → tensor umwandeln
    param_dict = {name: p for (name, _), p in zip(model.named_parameters(), params)}
    return functional_call(model, param_dict, (x,))
