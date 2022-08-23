#ifndef __DUMMY_BACKEND_HPP
#define __DUMMY_BACKEND_HPP
//#include<tmci/backend.hpp>
#include "/home/mmadhya1/tmci/tmci/src/backend.hpp" 
#include "model_client.hpp"
#include <string>
#include<list>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <limits>
#include<memory>
#define __DEBUG
#include "debug.hpp"


class DummyBackend : public tmci::Backend {

    std::unique_ptr<model_client_t> client;
    std::vector<uint64_t> profile_time_stamps;
    public:

	DummyBackend(const char* config) {
		std::vector<std::string>servers;
		std::vector<int>providers;
		getServers(servers, providers);
		client.reset(new model_client_t(servers, providers));
	}

    virtual inline int getTimeStamps(uint64_t *ts){
       for(int i=0; i<profile_time_stamps.size(); ++i){
           ts[i] = profile_time_stamps[i];
       }
       return profile_time_stamps.size();
    }

    virtual inline void clearTimeStamps(){
        profile_time_stamps.clear();
    }
    bool isGPUPtr(const void* ptr);

	void getServers(std::vector<std::string>&servers, std::vector<int>&providers);

    uint32_t Signed32ToUnsigned32(int32_t ele);

    uint64_t ConcatUnsigned32ToUnsigned64(uint32_t first, uint32_t second);

    std::vector<uint64_t> ConcatSigned32ToUnsigned64(std::vector<int32_t>&elements);

	virtual int Save(const std::list<std::reference_wrapper<const tensorflow::Tensor>>& tensors, std::vector<int32_t>&modelids_int32, std::vector<int32_t>&lids_int32);

	virtual int Load(const std::list<std::reference_wrapper<const tensorflow::Tensor>>& tensors, std::vector<int32_t>&modelids_int32, std::vector<int32_t>&lids_int32, std::vector<int32_t>&lowners_int32 );
	virtual bool store_meta(uint64_t id, uint64_t *edges, int m, uint64_t *lids, uint64_t *owners, uint64_t *sizes, int n);
	
	virtual int get_composition(uint64_t id, uint64_t *lids, uint64_t *owners, int n);

	virtual int get_prefix(uint64_t *edges, int n, uint64_t *id, uint64_t *result) ;

    virtual bool update_ref_counter(uint64_t id, int value); 

	};
#endif
