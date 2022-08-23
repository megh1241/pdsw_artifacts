from pathlib import Path
from typing import Optional, List, Tuple
from contextlib import contextmanager
import hashlib
import tensorflow as tf
import copy
import collections
import time
import redis
import json
import abc
import os
import ctypes
import redis.lock as rlock
from deephyper.nas.metrics import r2
import numpy as np
import uuid
import sys
import tmci.plugins
import tmci.checkpoint
import time
import math
from datetime import datetime
# Init ctypes types
DOUBLE = ctypes.c_double
PDOUBLE = ctypes.POINTER(DOUBLE)
PPDOUBLE = ctypes.POINTER(PDOUBLE)
PPPDOUBLE = ctypes.POINTER(PPDOUBLE)
def timestamp(dt):
    epoch = datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds() * 1000.0

class TransferMethod(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def transfer(self, model: tf.keras.Model, hint: Optional[str] = None) -> Tuple[tf.keras.Model, List[str]]:
        """transfer weights into the model, returning the transfer model and list of layer id's transfered"""
        pass

    @abc.abstractmethod
    def store(self, id:str, model: tf.keras.Model, prefix: List[str]) -> str:
        """store weights; empty prefix means store everything; returns the id of a model"""
        pass

    @abc.abstractmethod
    def retire_model(self, id: str):
        """removes a model and its weights"""
        pass

class DataStatesModelRepo(TransferMethod):
    def __init__(self):
        #Links to libmodel_client.so. This calls cpp-store/client-lib.cpp constructor ()
        #which finds all running servers (by reading the list from a text file and emplaces it onto servers vector
        #It then creates a model_client() instance.
        #Note: make sure this is called only once
        self.lib = tmci.plugins.load('libdummy.so')
        self.ancestors = {}
        self.to_hash = {}
        self.to_name = {}


    def __uint64_array(self, l):
        return (ctypes.c_uint64 * len(l))(*l)

    def __float_array(self, l):
        return (ctypes.c_float * len(l))(*l)

    def __ubyte_array(self, l):
        return (ctypes.POINTER(ctypes.c_ubyte) * len(l))(*l)

    def __np_as_ubyte(self, array):
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    def __double_array(self, l):
        return (ctypes.POINTER(ctypes.c_double) * len(l))(*l)

    def __np_as_double(self, array):
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


    def __2dlist_as_double_arr(self, n_rows, n_cols):
        a = []
        for i in range(n_rows):
            temp = np.zeros(n_cols, dtype=np.float64)
            a.append(temp)

        ctypes_arr = self.__double_array([self.__np_as_double(array) for array in a])
        return ctypes_arr


    def __double2ArrayToPointer(self, arr):

        # Init needed data types
        ARR_DIMX = DOUBLE*arr.shape[1]
        ARR_DIMY = PDOUBLE*arr.shape[0]

        # Init pointer
        arr_ptr = ARR_DIMY()

        # Fill the 2D ctypes array with values
        for i, row in enumerate(arr):
            arr_ptr[i] = ARR_DIMX()

        for j, val in enumerate(row):
            arr_ptr[i][j] = val


        return arr_ptr


    def __double2pointerToArray(self, ptr, n, m):
        arr = np.zeros(shape=(n, m))

        for i in range(n):
            for j in range(m):
                arr[i,j] = ptr[i][j]

        return arr

    def __hash_layers(self, model_cfg):
        for layer in model_cfg['layers']:
            #Two layers are shareable only if their configs match
            #Hash maps every config to a a unique value

            u = layer['name']
            if u in self.to_hash:
                continue
            h = int.from_bytes(hashlib.md5(u.encode()).digest()[12:], 'big')
            self.to_hash[u] = h
            self.to_name[h] = u

    def __extract_tensor_config(self, model, prefix):
        sizes = []
        l = []
        if not prefix:
            prefix = [layer.name for layer in model.layers]
        
        for layer in model.layers:
            if layer.name in prefix:
                h = self.to_hash[layer.name]
                num_weights = len(layer.weights)
                for i in range(num_weights):
                    l.append(np.uint64(h+i))
                    shp = layer.weights[i].get_shape()
                    dtype_size = layer.weights[i].dtype.size
                    prod = math.prod(shp) * dtype_size
                    sizes.append(prod)
        return l,sizes 

    def __config_to_edges(self, model_cfg):
        edges = []
        for layer in model_cfg['layers']:
            u = layer['name']
            for in_node in layer['inbound_nodes']:
                v = in_node[0][0]
                edges += [self.to_hash[v], self.to_hash[u]]
        return self.__uint64_array(edges)


    def store_meta(self, id, model_cfg):
        now = datetime.now()
        start_time_main = timestamp(now)
        self.__hash_layers(model_cfg)
        edges = self.__config_to_edges(model_cfg)
        lids = lsizes = lowners = self.__uint64_array([id] * len(model_cfg['layers']))
        now = datetime.now()
        end_time_main = timestamp(now)
        return self.lib.store_meta(ctypes.c_uint64(id), edges, len(edges), lids, lowners, lsizes, len(lids))


    def _best_match(self, model):
        return self.get_prefix(model.get_config())


    def get_prefix(self, model_cfg):
        self.__hash_layers(model_cfg)
        edges = self.__config_to_edges(model_cfg)
        cid = ctypes.c_uint64()
        result = self.__uint64_array([0] * len(edges))
        res_len = self.lib.get_prefix(edges, len(edges), ctypes.byref(cid), result)
        transferred = [self.to_name[result[i]] for i in range(res_len)]
        return (cid, transferred)



    def get_time_stamps(self):
        max_ts = 100
        c_ts = self.__uint64_array([0] * max_ts)
        n = self.lib.get_time_stamps(c_ts)
        return c_ts[0:n] 

    def clear_time_stamps(self):
        self.lib.clear_time_stamps()

    def store(self, id, model, prefix):
        cid = ctypes.c_uint64(id)
        #Hash the names of all the tensors.
        #Populates to_hash: from name to hash and to_name: from hash to name
        self.__hash_layers(model.get_config())
        lids,sizes = self.__extract_tensor_config(model, prefix)
        tot_size = sum(sizes)
        #TODO : exception handling
        
        tmci.checkpoint.save_weights(model,  prefix, backend='dummy', model_ids=np.uint64(id), lids=lids, config='.', include_optimizer=False)
        now = datetime.now()
        start_time_main = timestamp(now)
        if len(prefix) < len(model.layers):
            if id not in self.ancestors:
                raise Exception("cannot store partial model without ancestor")
            comp = copy.deepcopy(self.ancestors[id])
            for i in range(len(lids)):
                comp[lids[i]] = (id, sizes[i])
            lids = self.__uint64_array(list(comp.keys()))
            lsizes = self.__uint64_array([size for _, size in comp.values()])
            lowners = self.__uint64_array([owner for owner, _ in comp.values()])
        else:
            lids = self.__uint64_array(lids)
            lsizes = self.__uint64_array(sizes)
            lowners = self.__uint64_array([id] * len(lids))
        now = datetime.now()
        end_time_main = timestamp(now)

        edges = self.__config_to_edges(model.get_config())
        now = datetime.now()
        end_time_main2 = timestamp(now)
        self.lib.store_meta(cid, edges, len(edges), lids, lowners, lsizes, len(lids))
        #self.lib.update_ref_counter(cid, 1)
        return True


    def transfer(self, id, model, hint=None):
        self.__hash_layers(model.get_config())
        cid, prefix = self.get_prefix(model.get_config())

        if not prefix:
            comp = {}
            prefix_lids,sizes  = self.__extract_tensor_config(model, prefix)
            for i in range(len(prefix_lids)):
                comp[prefix_lids[i]] = (id, sizes[i])
            self.ancestors[id] = comp
            return [], None
       
        lids,sizes = self.__extract_tensor_config(model,prefix)
        tot_size = sum(sizes)
        lowners = self.__uint64_array([0] * len(lids))
        prefix_lids = self.__uint64_array(lids)
        self.lib.get_composition(cid, prefix_lids, lowners, len(prefix_lids))
        tmci.checkpoint.load_weights(model, prefix, backend='dummy', lowners=lowners[:],  model_ids=np.uint64(cid.value), lids=lids, config='.', include_optimizer=False)    
        comp = {}
        for i in range(len(prefix_lids)):
            comp[lids[i]] = (lowners[i], sizes[i])
        self.ancestors[id] = comp
        return prefix, cid




    def retire_model(self, id):
        cid = ctypes.c_uint64(id).value
        return self.lib.update_ref_counter(cid, -1)


def _standardize_names(model):
    # first build the graph of the layers
    graph = collections.defaultdict(list)
    hashed_names = collections.defaultdict(str)
    counts = collections.defaultdict(int)
    if isinstance(model, tf.keras.Sequential):
        #functional model doesn't have 'inbound_nodes', simply iterate an build graph linearly
        for last_layer, current_layer in zip(model.layers, model.layers[1:]):
            graph[current_layer.name] = [last_layer.name]
    elif isinstance(model, tf.keras.Model):
        #assume instance of functional model
        model_config = model.get_config()
        for layer in model_config['layers']:
            if len(layer['inbound_nodes']) > 0:
                graph[layer['name']] = [
                        inbound[0] for inbound in layer['inbound_nodes'][0]
                        ]

    def inner(layer):
        #we can't assume that layers aren't shared, so force a copy
        layer_config = copy.deepcopy(layer.get_config())

        #we need the old name to lookup predecessors
        old_name = layer_config['name']

        #we don't want to hash the name since this would make things recursively defined
        del layer_config['name']

        if 'trainable' in layer_config:
            layer_config['trainable'] = True
        begin = time.perf_counter()
        layer_hash = hashlib.sha3_512()
        layer_hash.update(json.dumps(layer_config, sort_keys=True).encode())
        for pred_name in graph[old_name]:
            layer_hash.update(hashed_names[pred_name].encode())
        layer_hash = layer_hash.hexdigest()
        end = time.perf_counter()

        base_name = layer.__class__.__name__ + layer_hash
        hashed_names[old_name] = base_name + "_" + str(counts[base_name])
        counts[base_name]+=1
        #create the new layer with the appropriate name
        layer_config['name'] = hashed_names[old_name]
        return layer.__class__.from_config(layer_config)
    return inner


def standardize_names(model: tf.keras.Model):
    return tf.keras.models.clone_model(model, input_tensors=model.inputs, clone_function=_standardize_names(model))
