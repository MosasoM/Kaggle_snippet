import numpy as np

def multi_logloss(ans,tar,only_label=False):
    # onlylabelはつまりonehotじゃなくてlabelencodingされてるかどうか
    if only_label:
        oh = np.identity(np.max(tar)+1)[tar]
    else:
        oh = tar
    ans_log = np.log(ans)
    return -np.sum(ans_log*oh)/len(ans)