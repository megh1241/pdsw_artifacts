#ifndef __MODEL_CLIENT
#define __MODEL_CLIENT

#include "model_utils.hpp"
#include <vector>
#include <string>
#include <map>
#include<chrono>
namespace tl = thallium;
using namespace std::chrono;
class model_client_t {
    tl::remote_procedure _store_meta, _get_prefix, _get_composition, _store_layers, _read_layers, _update_ref_counter;
    std::vector<tl::provider_handle> providers;
    std::unordered_map<model_id_t, composition_t> comp_cache;
    tl::mutex cache_lock;
    tl::engine engine;

public:
    inline tl::provider_handle &get_provider(model_id_t id) {
        return providers[id % providers.size()];
    }
    model_client_t(const std::vector<std::string> &servers, const std::vector<int>&provider_ids);
    bool store_meta(const digraph_t &g, const composition_t &comp, std::vector<uint64_t>&timestamps);
    prefix_t get_prefix(const digraph_t &child, std::vector<uint64_t>&timestamps);
    composition_t& get_composition(const model_id_t &id, std::vector<uint64_t>&timestamps);
    bool store_layers(const model_id_t &id, const vertex_list_t &layer_id,
                      std::vector<segment_t> &segments, std::vector<uint64_t>&timestamps);
                      //std::vector<std::pair<void*,size_t>> &segments);
    bool read_layers(const model_id_t &id, const vertex_list_t &layer_id, std::vector<segment_t>&segment_list, std::vector<uint64_t> &owners, std::vector<uint64_t>&timestamps);//std::vector<std::pair<void*,size_t>> &segments);
    bool update_ref_counter(const model_id_t &id, int value);
    bool clear_timestamps(const model_id_t &id);
    std::vector<uint64_t> get_timestamps(const model_id_t &id);
};

#endif
