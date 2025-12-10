from configs.loader import load_index

class IndexMemoryCalculator:
    def __init__(self, vlite_cfg, list_sizes=None):
        index_cfg = load_index()[vlite_cfg.index]
        self.ntotal = index_cfg['ntotal']
        self.nlist = index_cfg['nlist']
        self.d = index_cfg['d']
        self.M = index_cfg['M']
        self.ksub = index_cfg['ksub']
        self.msub = index_cfg['msub']
        self.list_sizes = list_sizes
        self.num_gpus = vlite_cfg.num_gpus
        self.partitioned = False
        
    def _centroid_mem(self, nnlist=None):
        if nnlist is None:
            nnlist = self.nlist
        return self.d * nnlist * 4 / (1024 ** 3)

    def _invlist_mem(self, nntotal):
        return nntotal * self.M * 0.5 / (1024 ** 3)

    def _ids_mem(self, nntotal):
        return nntotal * 4 / (1024 ** 3)

    def _pq_mem(self):
        return self.M * self.msub * self.ksub * 4 / (1024 ** 3)

    def get_total_size(self, ppt, partitioned=True, dedicated=False, n=1):
        nnlist = int(ppt * self.nlist)
        if self.list_sizes is not None:
            nntotal = sum(self.list_sizes[:nnlist])
        else:
            nntotal = self.ntotal

        centroid_mem = self._centroid_mem(nnlist)
        invlist_mem = self._invlist_mem(nntotal)
        ids_mem = self._ids_mem(nntotal)
        pq_mem = self._pq_mem()
        others = 2.0  # GB, fixed overhead
        
        if dedicated:
            return pq_mem + others + centroid_mem + (invlist_mem + ids_mem) / n

        if partitioned:
            return  pq_mem + others + (centroid_mem + invlist_mem + ids_mem) / self.num_gpus
        else:
            return centroid_mem + pq_mem + invlist_mem + ids_mem + others