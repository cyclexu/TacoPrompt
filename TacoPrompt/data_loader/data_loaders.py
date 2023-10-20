import torch
from torch.utils.data import DataLoader
from functools import partial

from .dataset import MAGDataset, RawDataset
from itertools import chain


class UnifiedDataLoader(DataLoader):
    def __init__(self, data_path, taxonomy_name, sampling_mode=1, batch_size=10, negative_size=20, max_pos_size=100,
                 expand_factor=50, shuffle=True, num_workers=8, cache_refresh_time=64,
                 tokenizer='/codes/b/offline_models/bert_model', test_topk=-1):
        self.sampling_mode = sampling_mode
        self.batch_size = batch_size
        self.negative_size = negative_size
        self.max_pos_size = max_pos_size
        self.expand_factor = expand_factor
        self.shuffle = shuffle
        self.cache_refresh_time = cache_refresh_time

        # raw_graph_dataset = MAGDataset(name=taxonomy_name, path=data_path, raw=False, existing_partition=True)
        raw_graph_dataset = MAGDataset(name=taxonomy_name, path=data_path, raw=True, existing_partition=False)
        self.dataset = RawDataset(raw_graph_dataset, sampling_mode=sampling_mode,
                                  negative_size=negative_size, max_pos_size=max_pos_size, expand_factor=expand_factor,
                                  cache_refresh_time=cache_refresh_time, test_topk=test_topk, tokenizer=tokenizer)
        self.tokenizer = self.dataset.tokenizer
        self.num_workers = num_workers
        super(UnifiedDataLoader, self).__init__(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                                collate_fn=self.collate_fn, num_workers=self.num_workers,
                                                pin_memory=True)
        self.n_samples = len(self.dataset)  # total number of samples that will be emitted by this data loader

    def collate_fn(self, samples):
        # input: q, p, c, b, w, (p,c) tag, p tag, c tag, b tag, w tag, p exists, c exists, b exists, w exists
        # qs, ps, cs, bs, ws, pct, pt, ct, bt, wt, pe, ce, be, we = map(list, zip(*chain(*samples)))
        #
        # # the _t suffix stands for tensor
        # qs_t, ps_t, cs_t, bs_t, ws_t = map(
        #     partial(self.tokenizer, return_tensors='pt', padding=True, truncation=True, max_length=64),
        #     [qs, ps, cs, bs, ws])
        # pct_t, pt_t, ct_t, bt_t, wt_t, pe_t, ce_t, be_t, we_t = map(torch.tensor, [pct, pt, ct, bt, wt, pe, ce, be, we])
        # return qs_t, ps_t, cs_t, bs_t, ws_t, pct_t, pt_t, ct_t, bt_t, wt_t, pe_t, ce_t, be_t, we_t

        # 加入了p, q, c的train idx
        qs, ps, cs, bs, ws, pct, pt, ct, bt, wt, pe, ce, be, we, pi, qi, ci = map(list, zip(*chain(*samples)))

        # the _t suffix stands for tensor
        qs_t, ps_t, cs_t, bs_t, ws_t = map(
            partial(self.tokenizer, return_tensors='pt', padding=True, truncation=True, max_length=64),
            [qs, ps, cs, bs, ws])
        pct_t, pt_t, ct_t, bt_t, wt_t, pe_t, ce_t, be_t, we_t, pit, qit, cit = \
            map(torch.tensor, [pct, pt, ct, bt, wt, pe, ce, be, we, pi, qi, ci])
        return qs_t, ps_t, cs_t, bs_t, ws_t, pct_t, pt_t, ct_t, bt_t, wt_t, pe_t, ce_t, be_t, we_t, pit, qit, cit

    def __str__(self):
        return "\n\t".join([
            f"sampling_mode: {self.sampling_mode}",
            f"batch_size: {self.batch_size}",
            f"negative_size: {self.negative_size}",
            f"expand_factor: {self.expand_factor}",
            f"cache_refresh_time: {self.cache_refresh_time}",
        ])
