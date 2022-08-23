#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <deque>
#include <vector>

std::ostream *logger = &std::cout;

#define __DEBUG
#include "debug.hpp"

typedef std::vector<uint64_t> prefix_t;
struct digraph_t {
    typedef std::unordered_set<uint64_t> vset_t;
    std::unordered_map<uint64_t, vset_t> out_edges;
    std::unordered_map<uint64_t, int> in_degree;
};

int match(uint64_t root, const digraph_t &parent, const digraph_t &child, prefix_t &prefix) {
    std::unordered_map<uint64_t, int> visits;
    std::deque<uint64_t> frontier{root};
    prefix.clear();
    while(frontier.size() > 0) {
        uint64_t u = frontier.front();
        frontier.pop_front();
        prefix.push_back(u);
        auto c_it = child.out_edges.find(u);
        if (c_it == child.out_edges.end())
            continue;
        for (auto const &v : c_it->second) {
            auto p_it = parent.out_edges.find(u);
            if (p_it != parent.out_edges.end() && p_it->second.count(v)) {
                visits[v]++;
                if (visits[v] == std::max(child.in_degree.at(v), parent.in_degree.at(v)))
                    frontier.push_back(v);
            }
        }
    }
    return prefix.size();
}

int main() {
    digraph_t g;
    std::vector<uint64_t> edges{0, 7, 0, 1, 1, 2, 1, 3, 1, 4, 2, 5, 3, 5, 4, 6, 5, 7, 5, 8, 6, 8, 8, 9};
    for (int i = 0; i < edges.size(); i += 2) {
        g.out_edges[edges[i]].insert(edges[i+1]);
        g.in_degree[edges[i+1]]++;
    }

    digraph_t g2;
    std::vector<uint64_t> edges2{0, 7, 0, 1, 1, 2, 1, 3, 1, 4, 2, 5, 3, 5, 4, 8, 5, 7, 5, 8, 8, 9};
    for (int i = 0; i < edges2.size(); i += 2) {
        g2.out_edges[edges2[i]].insert(edges2[i+1]);
        g2.in_degree[edges2[i+1]]++;
    }

    prefix_t result;
    match(0, g, g2, result);

    for (auto &i : result)
        std::cout << i <<  " ";
    std::cout << std::endl;

    return 0;
}
