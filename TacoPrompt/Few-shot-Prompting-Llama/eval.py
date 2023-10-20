import pickle

def rec_at_k(rank_lists, tot_gt_len, k=10):
    right = 0
    for ranks in rank_lists:
        if ranks[0] == -1:
            continue
        for r in ranks:
            if r <= k:
                right += 1
    return right / tot_gt_len


def hit_at_k(rank_lists, k=10):
    hit = 0
    for ranks in rank_lists:
        if ranks[0] == -1:
            continue
        for r in ranks:
            if r <= k:
                hit += 1
                break
    return hit / len(rank_lists)


def evaluate(pickle_path):
    print('loading pickled dataset')
    with open(pickle_path, "rb") as fin:
        data = pickle.load(fin)
        rank_lists = data["rank_lists"]
        leaf_rank_lists = data["leaf_rank_lists"]
        non_leaf_rank_lists = data["nonleaf_rank_lists"]
        gt_len = data["gt_len"]
    print("==============Total==============")
    print("R@1: {}\nR@5: {}\nR@10: {}\nH@1: {}\nH@5: {}\nH@10: {}".format(
        rec_at_k(rank_lists, gt_len, 1), rec_at_k(rank_lists, gt_len, 5), rec_at_k(rank_lists, gt_len, 10),
        hit_at_k(rank_lists, 1), hit_at_k(rank_lists, 5), hit_at_k(rank_lists, 10)
    ))


if __name__ == '__main__':
    pickle_path = "saved_rank_lists/food_rank_lists.pickle"
    evaluate(pickle_path)