def copy_state_dict(cur_state_dict, pre_state_dict, prefix = ''):
    def _get_params(key):
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None
    
    for k in cur_state_dict.keys():
        v = _get_params(k)
        try:
            if v is None:
                print('parameter {} not found'.format(k))
                continue
            cur_state_dict[k].copy_(v)
        except:
            print('copy param {} failed'.format(k))
            continue