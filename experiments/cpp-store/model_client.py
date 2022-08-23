import abc
import hashlib
import numpy as np
import ctypes
import copy

class TransferModelRepo(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def transfer(self, id: int, model) -> list[str]:
        """transfer weights into the model, returning the transfer model and list of layer id's transfered"""
        pass

    @abc.abstractmethod
    def store(self, id: int, model, prefix: list[str]) -> bool:
        """store weights; empty prefix means store everything; returns the id of a model"""
        pass

    @abc.abstractmethod
    def retire(self, id: int) -> bool:
        """removes a model and its weights"""
        pass

class DataStatesModelRepo(TransferModelRepo):
    def __init__(self):
        self.lib = ctypes.CDLL("./libmodel_client.so")
        self.ancestors = {}
        self.to_hash = {}
        self.to_name = {}

    def __uint64_array(self, l):
        return (ctypes.c_uint64 * len(l))(*l)

    def __ubyte_array(self, l):
        return (ctypes.POINTER(ctypes.c_ubyte) * len(l))(*l)

    def __np_as_ubyte(self, array):
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    def __hash_layers(self, model_cfg):
        for layer in model_cfg['layers']:
            u = layer['name']
            if u in self.to_hash:
                continue
            h = int.from_bytes(hashlib.md5(u.encode()).digest()[8:], 'big')
            self.to_hash[u] = h
            self.to_name[h] = u

    def __extract_tensors(self, model, prefix):
        wmap = {}
        w = []
        l = []
        for layer in prefix:
            h = self.to_hash[layer.name]
            wmap[h] = layer.get_weights()
            w += wmap[h]
            l += [h + i for i in range(len(wmap[h]))]
        prefix_lids = self.__uint64_array(l)
        prefix_lsizes = self.__uint64_array([array.nbytes for array in w])
        prefix_ptrs = self.__ubyte_array([self.__np_as_ubyte(array) for array in w])
        return (wmap, prefix_lids, prefix_lsizes, prefix_ptrs)

    def __config_to_edges(self, model_cfg):
        edges = []
        for layer in model_cfg['layers']:
            u = layer['name']
            for in_node in layer['inbound_nodes']:
                v = in_node[0][0]
                edges += [self.to_hash[v], self.to_hash[u]]
        return self.__uint64_array(edges)

    def store_meta(self, id, model_cfg):
        self.__hash_layers(model_cfg)
        edges = self.__config_to_edges(model_cfg)
        lids = lsizes = lowners = self.__uint64_array([id] * len(model_cfg['layers']))
        return self.lib.store_meta(ctypes.c_uint64(id).value, edges, len(edges), lids, lowners, lsizes, len(lids))

    def _best_match(self, model):
        return self.get_prefix(model.get_config()['config'])

    def get_prefix(self, model_cfg):
        self.__hash_layers(model_cfg)
        edges = self.__config_to_edges(model_cfg)
        cid = ctypes.c_uint64()
        result = self.__uint64_array([0] * len(edges))
        res_len = self.lib.get_prefix(edges, len(edges), ctypes.byref(cid), result)
        transferred = [self.to_name[result[i]] for i in range(res_len)]
        return (cid.value, transferred)

    def store(self, id, model, prefix):
        if not prefix:
            prefix = model.layers
        cid = ctypes.c_uint64(id).value
        self.__hash_layers(model.get_config())
        _, prefix_lids, prefix_lsizes, prefix_ptrs = self.__extract_tensors(model, prefix)
        if not self.lib.store_layers(cid, prefix_lids, prefix_lsizes, prefix_ptrs, ctypes.c_size_t(len(prefix_lids)).value):
            raise Exception("cannot store layers for model %lu" % cid)
        if len(prefix) < len(model.layers):
            if id not in self.ancestors:
                raise Exception("cannot store partial model without ancestor")
            comp = copy.deepcopy(self.ancestors[id])
            for i in range(len(prefix_lids)):
                comp[prefix_lids[i]] = (id, prefix_lsizes[i])
            lids = self.__uint64_array(list(comp.keys()))
            lsizes = self.__uint64_array([size for _, size in comp.values()])
            lowners = self.__uint64_array([owner for owner, _ in comp.values()])
        else:
            lids = prefix_lids
            lsizes = prefix_lsizes
            lowners = self.__uint64_array([id] * len(lids))
        edges = self.__config_to_edges(model.get_config())
        self.lib.store_meta(cid, edges, len(edges), lids, lowners, lsizes, len(lids))
        self.lib.update_ref_counter(cid, 1)
        return True

    def transfer(self, id, model):
        cid, prefix = self.get_prefix(model.get_config())
        if not prefix:
            return None
        wmap, prefix_lids, prefix_lsizes, prefix_ptrs = self.__extract_tensors(model, [model.get_layer(name) for name in prefix])
        if not self.lib.read_layers(cid, prefix_lids, prefix_ptrs, len(prefix_lids)):
            raise Exception("cannot read prefix from ancestor model: %lu", cid)
        for layer in model.layers:
            if layer.name not in prefix:
                continue
            layer.set_weights(wmap[self.to_hash[layer.name]])
        comp = {}
        for i in range(len(prefix_lids)):
            comp[prefix_lids[i]] = (cid, prefix_lsizes[i])
        self.ancestors[id] = comp
        return prefix

    def retire(self, id):
        cid = ctypes.c_uint64(id).value
        return self.lib.update_ref_counter(cid, -1)
