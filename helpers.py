def save_parameters(name, value, epoch):
    filename =  name.replace(':', '-colon-') + '-epoch-{}.txt'.format(epoch)
    if len(value.shape) == 1: # is a list
        string_form = ','.join([str(number) for number in value])
    else:
        string_form = '\n'.join([','.join([str(number) for number in value[row]])
                                 for row in range(value.shape[0])])

    with open("/saved_paras/" + filename, "w") as f:
        f.write(string_form)

def restore_parameters(name, epoch):
    filename = name.replace(":", "-colon-") + "-epoch-{}.txt".format(epoch)
    with open("/saved-paras/" + filename) as f:
        lines = f.read().splitlines()

    if len(lines) == 1:
        value = [float(number) for number in lines[0].split(',')]
    else:
        value = [[float(number) for number in lines[row].split(',')]
                 for row in range(len(lines))]

    return value