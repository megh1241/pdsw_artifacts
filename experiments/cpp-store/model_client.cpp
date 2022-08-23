#include "model_client.hpp"
#include <functional>

#define __DEBUG
#include "debug.hpp"
using namespace std::chrono;

model_client_t::model_client_t(const std::vector<std::string> &servers, const std::vector<int>&provider_ids) :
    engine("ofi+verbs", THALLIUM_CLIENT_MODE) {
    _store_meta = engine.define("store_meta");
    _get_prefix = engine.define("get_prefix");
    _get_composition = engine.define("get_composition");
    _store_layers = engine.define("store_layers");
    _read_layers = engine.define("read_layers");
    _update_ref_counter = engine.define("update_ref_counter");
    int iter=0;
    for (auto &server : servers) {
        tl::endpoint endp = engine.lookup(server);
        providers.emplace_back(tl::provider_handle(endp, provider_ids[iter]));
        iter++;
    }
}

bool model_client_t::store_meta(const digraph_t &g, const composition_t &comp, std::vector<uint64_t>&profile_time_stamps) {
    timestamp_t ret = _store_meta.on(get_provider(g.id))(g, comp );
    profile_time_stamps.emplace_back(ret.first);
    profile_time_stamps.emplace_back(ret.second);
    return true;
}

bool model_client_t::store_layers(const model_id_t &id, const vertex_list_t &layer_id,
					std::vector<segment_t>&segments, std::vector<uint64_t>&profile_time_stamps){

    std::vector<size_t> layer_size;   
    for (auto &segment: segments)
        layer_size.emplace_back(segment.second);

    tl::bulk bulk = engine.expose(segments, tl::bulk_mode::read_write);
    std::vector<uint64_t> ts = _store_layers.on(get_provider(id))(id, layer_id, layer_size, bulk);
    profile_time_stamps.insert(profile_time_stamps.end(), ts.begin(), ts.end());
    return true;
}

composition_t &model_client_t::get_composition(const model_id_t &id, std::vector<uint64_t>&profile_time_stamps) {
    std::unique_lock<tl::mutex> lock(cache_lock);

    auto it = comp_cache.find(id);
    std::pair<composition_t, timestamp_t> comp_result = _get_composition.on(get_provider(id))(id, profile_time_stamps);
    if (it == comp_cache.end())
        it = comp_cache.emplace_hint(it, id, comp_result.first);
    else if (it->second.size() == 0){
        it->second = comp_result.first;
    }
    profile_time_stamps.emplace_back(comp_result.second.first);
    profile_time_stamps.emplace_back(comp_result.second.second);
    return it->second;
}

bool model_client_t::read_layers(const model_id_t &id, const vertex_list_t &layer_id, std::vector<segment_t>&segment_list, std::vector<uint64_t> &owners, std::vector<uint64_t>&profile_time_stamps){// std::vector<std::pair<void*,size_t>> &segments2) {
    //Layer_id is a list of tensor ids (layer hash + tensor number) 
    //uint64_t ts_begin = duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
    
    struct req_info_t {
        vertex_list_t layer_id;
        //segment_t : For bulk RDMA
        std::vector<segment_t> segments;
    };
    std::unordered_map<model_id_t, req_info_t> owner_map;
    std::vector<tl::bulk> bulks;
    std::vector<tl::async_response> reps;

   
    for (int i = 0; i < layer_id.size(); i++) {
        auto owner = owners[i];
        auto &e = owner_map[owner];
        e.layer_id.emplace_back(layer_id[i]);
        e.segments.emplace_back(segment_list[i]);
    }
    for (auto &e : owner_map) {
        bulks.emplace_back(engine.expose(e.second.segments, tl::bulk_mode::write_only));
        reps.emplace_back(_read_layers.on(get_provider(e.first)).async(e.second.layer_id, e.first, bulks.back()));
    }
    bool result = true;
    for (auto &rep : reps){
        std::vector<uint64_t> temp = rep.wait();
        profile_time_stamps.insert(profile_time_stamps.end(), temp.begin(), temp.end());
    }
    return result;
}

bool model_client_t::update_ref_counter(const model_id_t &id, int value) {
    std::unordered_map<model_id_t, vertex_list_t> req_args;
    std::vector<tl::async_response> reps;
    std::vector<uint64_t> temp;
    auto &comp = get_composition(id, temp);
    if (comp.empty())
        return false;
    for (auto &e : comp)
        req_args[e.second.first].emplace_back(e.first);
    for (auto &e : req_args)
        reps.push_back(_update_ref_counter.on(get_provider(id)).async(e.first, e.second, value));
    bool result = true;
    for (auto &rep : reps)
        result = result && rep.wait();
    return result;
}

prefix_t model_client_t::get_prefix(const digraph_t &child, std::vector<uint64_t>&profile_time_stamps) {
    prefix_t max_result;
    std::vector<tl::async_response> requests;
    for (auto &provider : providers)
        requests.emplace_back(_get_prefix.on(provider).async(child));
    for (auto &request : requests) {
        std::pair<prefix_t, timestamp_t> result = request.wait();
        if (result.first.second.size() > max_result.second.size())
            std::swap(result.first, max_result);
        profile_time_stamps.push_back(result.second.first); 
        profile_time_stamps.push_back(result.second.second); 
    }
    return max_result;
}

