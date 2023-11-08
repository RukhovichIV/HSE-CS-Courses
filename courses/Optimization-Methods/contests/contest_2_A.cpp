#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

void dfs_tout(const std::vector<std::vector<uint16_t>>& edges,
              std::vector<uint8_t>& was, std::vector<uint16_t>& order,
              uint16_t v) {
    was[v] = 1u;
    for (uint16_t i = 0; i < edges[v].size(); ++i) {
        if (!was[edges[v][i]]) dfs_tout(edges, was, order, edges[v][i]);
    }
    order.push_back(v);
}

void dfs_comp(const std::vector<std::vector<uint16_t>>& edges,
              std::vector<uint16_t>& comp, uint16_t v, uint16_t com) {
    comp[v] = com;
    for (uint16_t i = 0; i < edges[v].size(); ++i) {
        if (!comp[edges[v][i]]) dfs_comp(edges, comp, edges[v][i], com);
    }
}

int main()
{
    std::cin.tie(0), std::cout.tie(0), std::ios_base::sync_with_stdio(0);
// #define file
#ifdef file
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif
    uint16_t n, m;
    std::cin >> n >> m;
    uint16_t vert_count = 2 * n;
    std::vector<std::vector<uint16_t>> edges(vert_count);
    std::vector<std::vector<uint16_t>> edgest(vert_count);
    std::vector<uint8_t> was(vert_count, 0u);
    std::vector<uint16_t> comp(vert_count, 0u);
    std::vector<uint16_t> order;
    for (uint16_t i = 0; i < m; ++i) {
        int16_t a, b;
        std::cin >> a >> b;
        a = (a > 0 ? (a - 1) * 2 : (-a - 1) * 2 + 1);
        b = (b > 0 ? (b - 1) * 2 : (-b - 1) * 2 + 1);

        edges[a^1].push_back(b);
        edges[b^1].push_back(a);
        edgest[b].push_back(a^1);
        edgest[a].push_back(b^1);
    }

    for (uint16_t v = 0; v < vert_count; ++v) {
        if (!was[v]) dfs_tout(edges, was, order, v);
    }

    uint16_t com = 1u;
    for (auto v = order.rbegin(); v != order.rend(); ++v) {
        if (!comp[*v]) dfs_comp(edgest, comp, *v, com++);
    }

    for (uint16_t v = 0; v < vert_count; v += 2) {
        if (comp[v] == comp[v^1]) {
            std::cout << "no";
            return 0;
        }
    }
    std::cout << (comp[0] > comp[1] ? "yes" : "no");
    return 0;
}
