#include "model_server.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <cxxopts.hpp>
#define __DEBUG
#include "debug.hpp"

std::ostream *logger = &std::cout;

int main(int argc, char **argv) {
    cxxopts::Options options("server", "One line description of MyProgram");

    options.add_options()
        ("thallium_connection_string", "Thallium connection string (eg ofi+verbs)",  cxxopts::value<std::string>()->default_value("ofi+verbs")) 
        ("num_threads", "Number of argobots threads for handling RPC requests", cxxopts::value<int>()->default_value("8"))
        ("provider_id", "Id of Thallium provider", cxxopts::value<int>()->default_value("0"))
        ("storage_backend", "rocksdb/map", cxxopts::value<std::string>()->default_value("rocksdb"))
        ("rocksdb_config", "large_memtable_pfs/default_pfs/default_tempfs", cxxopts::value<std::string>()->default_value("default_pfs"))
        ("server_full_address_filename", "Path of the file that contains the server address", cxxopts::value<std::string>()->default_value("/home/mmadhya1/experiments/cpp-store/server_str.txt"))
        ;

    auto result = options.parse(argc, argv);

    auto thallium_connection_string = result["thallium_connection_string"].as<std::string>();
    auto num_threads = result["num_threads"].as<int>();
    auto provider_id = result["provider_id"].as<int>();
    auto storage_backend = result["storage_backend"].as<std::string>();
    auto rocksdb_config = result["rocksdb_config"].as<std::string>();
    auto server_fname = result["server_full_address_filename"].as<std::string>();
    std::cout<<"filename; "<<server_fname<<"\n";
    fflush(stdout);

    tl::engine engine(thallium_connection_string.c_str(), THALLIUM_SERVER_MODE);
    std::fstream fout;
    fout.open(server_fname.c_str(), std::ios::out|std::ios::app);
    fout<<engine.self()<<"\n"<<provider_id<<"\n"<<std::flush;
    fout.close();
    model_server_t provider(engine, provider_id, num_threads, storage_backend, rocksdb_config);
    return 0;
}
