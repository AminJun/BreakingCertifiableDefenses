import os
import datetime
from args import Args


def _load(file_address: str) -> int:
    with open(file_address, 'r') as f:
        for l in f:
            return int(l.split(' ')[0])


def _save(file_address: str, cur: int):
    with open(file_address, 'w') as f:
        f.write(str(cur + 1))


def get_exp_no() -> int:
    file_address = os.path.join(os.path.split(__file__)[0], 'exp_stat.db')
    output = _load(file_address)
    _save(file_address, output)
    return output


def _two_digits(cur: str) -> str:
    return '0' + cur if len(cur) == 1 else cur


def _time():
    now = datetime.datetime.now()
    arr = [now.day, now.hour, now.minute, str(now.second)[0]]
    return ''.join([_two_digits(str(x)) for x in arr])


def get_exp_name() -> str:
    args = Args()
    name_list = [str(get_exp_no()), _time()]
    exp_name = args.get_args().experiment
    if str(exp_name) is not 'None':
        name_list.append(exp_name)
    name_list.append(args.get_name())
    return '_'.join(name_list)
