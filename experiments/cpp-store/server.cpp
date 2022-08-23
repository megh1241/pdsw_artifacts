#include "model_server.hpp"

#include <iostream>
#include <fstream>
#include <string>
#define __DEBUG
#include "debug.hpp"
#include <argparse.hpp>

std::ostream *logger = &std::cout;

int main(int argc, char **argv) {
    /*argparse::ArgumentParser parser("server");
    parser.add_argument("--thallium_connection_string")
        .help("communication protocol");

    parser.add_argument("--num_threads")
        .default_value(std::thread::hardware_concurrency())
        .help("Number of argobots threads for handling RPC requests");

    parser.add_argument("--provider_id")
        .default_value(0)
        .help("Id of Thallium provider");

    parser.add_argument("--storage_backend")
        .default_value("rocksdb")
        .help("rocksdb/map");

    parser.add_argument("--rocksdb_config")
        .default_value("large_memtable")
        .help("large_memtable_pfs/default_pfs/default_tempfs");

    parser.add_argument("--server_full_address_filename")
        .default_value(std::string("/home/mmadhya1/experiments/cpp-store/server_str.txt"))
        .help("Path of the file that contains the server address");

    try {
        parser.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cout<<"error!!!1\n";
       fflush(stdout); 
       std::cerr << err.what() << std::endl;
        std::cerr << parser;
        std::exit(1);
    }

    std::string thallium_connection_string = parser.get<std::string>("--thallium_connection_string");
    int num_threads = parser.get<int>("--num_threads");
    int provider_id = parser.get<int>("--provider_id");
    std::string storage_backend = parser.get<std::string>("--storage_backend");
    std::string rocksdb_config = parser.get<std::string>("--large_memtable");
    std::string server_fname = parser.get<std::string>("--server_full_address_filename");
       std::cout<<"filename; "<<server_fname<<"\n";
       fflush(stdout);
    */
    std::string thallium_connection_string = std::string("ofi+verbs");
    int num_threads = 8;
    int provider_id = 0; 
    std::string storage_backend = std::string("rocksdb");
    std::string rocksdb_config  = std::string("large_memtable");
    std::string server_fname = std::string("/home/mmadhya1/experiments/cpp-store/server_str.txt");
    std::cout<<"filename; "<<server_fname<<"\n";
    fflush(stdout);

    tl::engine engine(thallium_connection_string.c_str(), THALLIUM_SERVER_MODE);
    std::fstream fout;
    fout.open(server_fname.c_str(), std::ios::out|std::ios::app);
    fout<<engine.self()<<"\n"<<provider_id<<"\n"<<std::flush;
    fout.close();
    model_server_t provider(engine, provider_id, num_threads, storage_backend);

    return 0;
}
