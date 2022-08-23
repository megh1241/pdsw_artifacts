#include "model_server.hpp"

#include <thread>
#include <deque>
#include <chrono>
#define __DEBUG
#include "debug.hpp"
#include <rocksdb/compaction_filter.h>
#include <rocksdb/db.h>
#include <rocksdb/advanced_options.h>
#include <rocksdb/table.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/iostats_context.h>
#include <rocksdb/slice_transform.h>
#include <rocksdb/perf_context.h>
#include <rocksdb/memtablerep.h>
using ROCKSDB_NAMESPACE::DB;
using ROCKSDB_NAMESPACE::OptimisticTransactionDB;
using ROCKSDB_NAMESPACE::OptimisticTransactionOptions;
using ROCKSDB_NAMESPACE::Options;
using ROCKSDB_NAMESPACE::ReadOptions;
using ROCKSDB_NAMESPACE::BlockBasedTableOptions;
using ROCKSDB_NAMESPACE::Snapshot;
using ROCKSDB_NAMESPACE::Status;
using ROCKSDB_NAMESPACE::Transaction;
using ROCKSDB_NAMESPACE::WriteOptions;
using ROCKSDB_NAMESPACE::NewBloomFilterPolicy;


using namespace std;
timestamp_t model_server_t::store_meta(const digraph_t &g, const composition_t &comp) {
    uint64_t ts_begin = duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
    std::unique_lock<tl::mutex> lock(store_lock);
    graph_store.emplace_back(g);
    graph_info.try_emplace(g.id, model_info_t(std::prev(graph_store.end()), comp));
    uint64_t ts_end = duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
    return std::make_pair(ts_begin, ts_end);
}

bool model_server_t::update_ref_counter(const model_id_t &owner, const vertex_list_t &layer_id, int value) {
    for (int i = 0; i < layer_id.size(); i++) {
        std::unique_lock<tl::mutex> lock(store_lock);
        auto &li = layer_store[layer_id[i]];
        lock.unlock();
        lock = std::unique_lock<tl::mutex>(li.layer_lock);
        auto it = li.owner_map.find(owner);
        if (it == li.owner_map.end())
            return false;
        it->second.ref_count += value;
        if (it->second.ref_count <= 0) {
            delete (unsigned char *)it->second.segment.first;
            li.owner_map.erase(it);
        }
    }
    if (value < 0) {
        std::unique_lock<tl::mutex> lock(store_lock);
        auto it = graph_info.find(owner);
        if (it != graph_info.end()) {
            graph_store.erase(it->second.index);
            graph_info.erase(it);
            DBG("retired model " << owner);
        }
    }
    return true;
}



void model_server_t::store_layers_rocksdb(std::vector<rocksdb::Slice>& value_slices,  std::vector<std::string> &keys, std::vector<uint64_t> &timestamps){
    rocksdb::WriteOptions write_options;
    write_options.disableWAL  = true;
    std::vector<uint64_t> rdb_timings(2, 0);
    size_t num_elements = value_slices.size();
    for(size_t i = 0; i < num_elements; i++) {
        rocksdb::SetPerfLevel(rocksdb::PerfLevel::kEnableTimeExceptForMutex); 
        rocksdb::get_perf_context()->Reset();
        rocksdb::get_iostats_context()->Reset();
        auto status = db->Put(write_options,  rocksdb::Slice(keys[i]), value_slices[i]);
        assert(status.ok());

        uint64_t t1 = rocksdb::get_perf_context()->write_memtable_time/1000;
        uint64_t t2 = rocksdb::get_perf_context()->write_delay_time/1000;
        
        rdb_timings[0]+=t1;
        rdb_timings[1]+=t2;
        rocksdb::SetPerfLevel(rocksdb::PerfLevel::kDisable);
    }
    uint64_t cum_sum = timestamps[timestamps.size()-1];
    for(auto &ele: rdb_timings){
        cum_sum += ele;
        timestamps.push_back(cum_sum);
    }
}
void model_server_t::store_layers(const tl::request &req, const model_id_t &id, const vertex_list_t &layer_id,
        const std::vector<size_t> &layer_size, tl::bulk &bulk){
    std::vector<uint64_t> timestamps;
    std::vector<rocksdb::Slice> slices;  
    std::vector<segment_t> segments;
    std::vector<layer_t> layers;
    std::vector<std::string> keys;
    size_t total_size=0;
    for (int i = 0; i < layer_id.size(); i++) {
        total_size+=layer_size[i];
        char *buffer = new char[layer_size[i]];
        segments.emplace_back((void*)buffer, layer_size[i]);
    }
    timestamps.push_back(duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count());
    tl::endpoint ep = req.get_endpoint();
    tl::bulk local = get_engine().expose(segments, tl::bulk_mode::read_write);
    timestamps.push_back(duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count());
    for(auto i=0; i<total_size; i+=rdma_transfer_size){
        auto chunk = std::min(rdma_transfer_size, total_size-i);
        bulk(i, chunk).on(ep) >> local(i, chunk);
    }
    timestamps.push_back(duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count());
    if (policy.compare(std::string("rocksdb")) == 0){
        int iter=0;
        for(auto &i: segments){
            std::string key = std::to_string(id) + std::to_string(layer_id[iter]);
            keys.emplace_back(key);
            slices.emplace_back(rocksdb::Slice((const char*)i.first, i.second)) ; 
            iter++;
        }
        store_layers_rocksdb(slices, keys, timestamps);
        for(auto &i: segments)
            delete (unsigned char *)i.first;
    }else{
        int iter=0;
        for(auto &i: segments){
            layers.emplace_back(layer_t(layer_size[iter]));
            std::memcpy(layers.back().segment.first, i.first, i.second);
            delete (unsigned char*)i.first;
            iter++;
        } 
        store_layers_map(id, layer_id, layer_size, layers);
    }
    timestamps.push_back(duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count());
    req.respond(timestamps);
}
void model_server_t::store_layers_map(const model_id_t &id, const vertex_list_t &layer_id,
        const std::vector<size_t> &layer_size, std::vector<layer_t> &layers) {

    for (int i = 0; i < layer_id.size(); i++) {
        std::unique_lock<tl::mutex> lock(store_lock);
        auto &li = layer_store[layer_id[i]];
        lock.unlock();
        lock = std::unique_lock<tl::mutex>(li.layer_lock);
        auto it = li.owner_map.find(id);
        if (it != li.owner_map.end()) {
            auto ptr = it->second.segment.first;
            it->second.segment = layers[i].segment;
            delete (unsigned char *)ptr;
        } else{
            li.owner_map.emplace_hint(it, id, layers[i]);
        }
    }
}

size_t model_server_t::read_layers_map(const vertex_list_t &layer_id,
        const model_id_t &owner, std::vector<segment_t>&segments){
    size_t total_size=0;
    for (int i = 0; i < layer_id.size(); i++) {
        auto &lid = layer_id[i];
        std::unique_lock<tl::mutex> lock(layer_store[lid].layer_lock);
        auto it = layer_store[lid].owner_map.find(owner);

        if (it != layer_store[lid].owner_map.end()){
            total_size += it->second.segment.second;
            segments.emplace_back(it->second.segment);
        }else{
            DBG("not found!");
        }
    }
    return total_size;

}
size_t model_server_t::read_layers_rocksdb(const vertex_list_t &layer_id,
        const model_id_t &owner, std::vector<segment_t>&segments, std::vector<rocksdb::PinnableSlice> &value_vec, std::vector<uint64_t> &timestamps){
    size_t total_size = 0;
    std::vector<uint64_t> rdb_timings(2, 0);
    for(int i=0; i<layer_id.size(); ++i){
        std::string key_str = std::to_string(owner) + std::to_string(layer_id[i]);
        const rocksdb::Slice key{key_str};
        rocksdb::SetPerfLevel(rocksdb::PerfLevel::kEnableTimeExceptForMutex);
        rocksdb::get_perf_context()->Reset();
        rocksdb::get_iostats_context()->Reset();
        auto status = db->Get(rocksdb::ReadOptions(), db->DefaultColumnFamily(), key, &value_vec[i]);
        assert(status.ok());
        rocksdb::SetPerfLevel(rocksdb::PerfLevel::kDisable);
        uint64_t t1 = rocksdb::get_perf_context()->block_read_time/1000;
        uint64_t t4 = rocksdb::get_perf_context()->get_from_memtable_time/1000;
      
        rdb_timings[0]+=t1;
        rdb_timings[1]+=t4;
        total_size += value_vec[i].size();

        segment_t segment{(char*)value_vec[i].data(), value_vec[i].size()};
        segments.emplace_back(segment);
        uint64_t time4 = duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
    }
    uint64_t cum_sum = timestamps[timestamps.size()-1];
    for(auto ele: rdb_timings){
        cum_sum += ele;
        timestamps.push_back(cum_sum);
    }

    return total_size;
}

void  model_server_t::read_layers(const tl::request &req, const vertex_list_t &layer_id,
        const model_id_t &owner, tl::bulk &layer_bulk) {

    std::vector<uint64_t> timestamps;
    size_t total_size;
    std::vector<segment_t> segments;
    std::vector<rocksdb::PinnableSlice> value_vec(layer_id.size());
    timestamps.push_back(duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count());
    if (policy.compare(std::string("rocksdb")) == 0)
        total_size = read_layers_rocksdb(layer_id, owner, segments, value_vec, timestamps); 
    else
        total_size = read_layers_map(layer_id, owner, segments); 

    timestamps.push_back(duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count());
    tl::bulk local = get_engine().expose(segments, tl::bulk_mode::read_write);
    tl::endpoint ep = req.get_endpoint();
    timestamps.push_back(duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count());

    for(auto i=0; i<total_size; i+=rdma_transfer_size){
        auto chunk = std::min(rdma_transfer_size, total_size-i);
        layer_bulk(i, chunk).on(ep) << local(i, chunk);
    }
    for(auto &value: value_vec){
        value.Reset();
    }
    timestamps.push_back(duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count());
    req.respond(timestamps);
    //req.respond(segments.size() == layer_id.size());
}

std::pair<composition_t, timestamp_t> model_server_t::get_composition(const model_id_t &id) {
    std::unique_lock<tl::mutex> lock(store_lock);
    timestamp_t ts;
    uint64_t ts_begin = duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
    auto it = graph_info.find(id);
    if (it == graph_info.end()){
        uint64_t ts_end = duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
        ts = std::make_pair(ts_begin, ts_end); 
        return std::make_pair(composition_t(), ts);
    }
    else{
        uint64_t ts_end = duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
        ts = std::make_pair(ts_begin, ts_end); 
        return std::make_pair(it->second.composition, ts);
    }
}

std::pair<prefix_t, timestamp_t> model_server_t::get_prefix(const digraph_t &child) {
    uint64_t ts_begin = duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
    vertex_list_t max_prefix;
    uint64_t max_id = 0;
    //TODO: We might need a lock for parent since graph_store is a shared resource that
    //may get updated by multiple threads simultaneously.
    for (auto &parent : graph_store) {
        std::deque<vertex_t> frontier{child.root};
        std::unordered_map<vertex_t, int> visits;
        vertex_list_t prefix;

        while(frontier.size() > 0) {
            uint64_t u = frontier.front();
            frontier.pop_front();
            prefix.push_back(u);
            auto c_it = child.out_edges.find(u);
            if (c_it == child.out_edges.end())
                continue;
            auto p_it = parent.out_edges.find(u);
            if (p_it == parent.out_edges.end())
                continue;
            for (auto const &v : c_it->second)
                if (p_it->second.count(v)) {
                    visits[v]++;
                    if (visits[v] == std::max(child.in_degree.at(v), parent.in_degree.at(v)))
                        frontier.push_back(v);
                }
        }
        if (prefix.size() > max_prefix.size()) {
            std::swap(prefix, max_prefix);
            max_id = parent.id;
        }
    }
    uint64_t ts_end = duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
    timestamp_t ts = std::make_pair(ts_begin, ts_end);
    return std::make_pair(std::make_pair(max_id, max_prefix), ts);
}


void model_server_t::init_rocksdb(uint16_t provider_id){
    rocksdb::Options options;
    options.compression = rocksdb::CompressionType::kNoCompression;
    options.create_if_missing = true;
    options.max_background_compactions = 4;
    options.max_background_flushes = 2;
    options.use_direct_reads = true;
    options.use_direct_io_for_flush_and_compaction = true;
    options.compaction_pri = rocksdb::CompactionPri::kMinOverlappingRatio;
    options.create_if_missing = true;
    
    uint64_t block_size = 1024;
    uint64_t t = 20;
    uint64_t cache_size = block_size * block_size * block_size * t;
    
    /*options.write_buffer_size = cache_size ; 
    BlockBasedTableOptions table_options;
    auto cache = rocksdb::NewLRUCache(cache_size); 
    table_options.block_cache = cache;
    
    table_options.cache_index_and_filter_blocks = true;
    table_options.block_size = cache_size / t;
    table_options.pin_l0_filter_and_index_blocks_in_cache = true;
    auto table_factory = NewBlockBasedTableFactory(table_options);
    options.table_factory.reset(table_factory);
    options.level_compaction_dynamic_level_bytes = true;
    */
    //options.env->SetBackgroundThreads(6, rocksdb::Env::Priority::HIGH);
    //options.env->SetBackgroundThreads(4, rocksdb::Env::Priority::LOW); 
    //std::string db_filename("/dev/shm/meghproov1");
    std::string root_dir("/home/mmadhya1/");
    //std::string root_dir("/dev/shm/");
    auto db_filename = root_dir + std::string("meghaanardb") + std::to_string(provider_id);
    auto status = rocksdb::DB::Open(options, db_filename, &db);
    assert(status.ok());   
}


model_server_t::model_server_t(tl::engine& e, uint16_t provider_id, uint32_t num_procs, std::string server_policy)
        : tl::provider<model_server_t>(e, provider_id), request_pool(tl::pool::create(tl::pool::access::spmc)) {
            self_provider_id = provider_id;
            policy=server_policy;
            if (policy.compare(std::string("rocksdb")) == 0){
                init_rocksdb(provider_id);
            }
            unsigned int n = std::thread::hardware_concurrency();
            std::cout << n << " concurrent threads are supported.\n";
            for(int i = 0; i < num_procs; i++)
                ess.emplace_back(tl::xstream::create(tl::scheduler::predef::deflt, *request_pool));

            define("store_meta", &model_server_t::store_meta, *request_pool);
            define("get_prefix", &model_server_t::get_prefix, *request_pool);
            define("get_composition", &model_server_t::get_composition, *request_pool);
            define("store_layers", &model_server_t::store_layers, *request_pool);
            define("read_layers", &model_server_t::read_layers, *request_pool);
            define("update_ref_counter", &model_server_t::update_ref_counter, *request_pool);
            DBG("thallium backend listening at: " << get_engine().self());
        }

    model_server_t::~model_server_t() {
        get_engine().wait_for_finalize();
        for (int i = 0; i < ess.size(); i++)
            ess[i]->join();
    }
