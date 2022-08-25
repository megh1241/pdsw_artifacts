#ifndef __MODEL_SERVER
#define __MODEL_SERVER

#include <utility>
#include <map>
#include <thread>
#include <chrono>
#include <rocksdb/db.h>
#include "model_utils.hpp"
#include <cassert>
#include<string>

#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <rocksdb/utilities/transaction.h>
#include <rocksdb/utilities/optimistic_transaction_db.h>

namespace tl = thallium;
using namespace std::chrono;

class model_server_t : public tl::provider<model_server_t> {
    struct model_info_t {
        std::vector<digraph_t>::iterator index;
        composition_t composition;
        model_info_t(const std::vector<digraph_t>::iterator &idx, const composition_t &comp) : index(idx), composition(comp) { }
    };
    struct layer_t {
        segment_t segment;
        size_t ref_count;
        //TODO: aadd access statistic
        layer_t(size_t size) : segment{new char[size], size}, ref_count(0) {}
    };
    struct layer_info_t {
        tl::mutex layer_lock;
        std::unordered_map<model_id_t, layer_t> owner_map;
    };
    std::string policy;
    uint16_t self_provider_id; 
    rocksdb::DB* db;
    std::vector<tl::managed<tl::xstream>> ess;
    tl::managed<tl::pool> request_pool;
    std::vector<digraph_t> graph_store;
    std::unordered_map<uint64_t, model_info_t> graph_info;
    std::unordered_map<vertex_t, layer_info_t> layer_store;
    tl::mutex store_lock;
public:
    model_server_t(tl::engine& e, uint16_t provider_id = 0, uint32_t num_procs=1, std::string const& server_policy=std::string("map"), std::string const& rocksdb_config=std::string("default_pfs"));
    ~model_server_t();
    timestamp_t store_meta(const digraph_t &g, const composition_t &comp );
    std::pair<prefix_t, timestamp_t> get_prefix(const digraph_t &child );
    std::pair<composition_t, timestamp_t> get_composition(const model_id_t &id);
    void store_layers(const tl::request &req, const model_id_t &id, const vertex_list_t &layer_id,
                      const std::vector<size_t> &layer_size, tl::bulk &layer_bulk);
    void store_layers_map(const model_id_t &id, const vertex_list_t &layer_id,
                    const std::vector<size_t> &layer_size, std::vector<layer_t> &layers);
    void store_layers_rocksdb(std::vector<rocksdb::Slice> &value_slices, std::vector<std::string> &keys, std::vector<uint64_t> &timestamps);
    void read_layers(const tl::request &req, const vertex_list_t &layer_id,
                     const model_id_t &owner, tl::bulk &layer_bulk);
    size_t read_layers_map( const vertex_list_t &layer_id,
                     const model_id_t &owner, std::vector<segment_t>&segments);
    size_t read_layers_rocksdb(const vertex_list_t &layer_id,
                            const model_id_t &owner, std::vector<segment_t>&segments, std::vector<rocksdb::PinnableSlice> &value_vec, std::vector<uint64_t> &timestamps);
    size_t read_layers_rocksdb_multiget(const vertex_list_t &layer_id,
                    const model_id_t &owner, std::vector<segment_t>&segments, std::vector<uint64_t> &timestamps);
    bool update_ref_counter(const model_id_t &owner, const vertex_list_t &layer_id, int value);
    bool clear_timestamps(const model_id_t &id);
    void init_rocksdb(uint16_t provider_id, std::string const& rocksdb_config);
    //void set_rocksdb_params(uint16_t provider_id);
    std::vector<uint64_t> get_timestamps(const model_id_t &id);
};

#endif //__MODEL_SERVER
