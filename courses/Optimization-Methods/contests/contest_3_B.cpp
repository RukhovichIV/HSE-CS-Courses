#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

class InputData {
public:
    uint16_t get_num_ones() {
        uint16_t ans = 0u;
        for (auto& elem : in) {
            ans += elem;
        }
        return ans;
    }

    std::vector<uint16_t> in;
    uint16_t out;
};

bool check_order(const InputData& fi, const InputData& se) {
    for (uint16_t i = 0; i < fi.in.size(); ++i) {
        if (fi.in[i] > se.in[i]) {
            return false;
        }
    }
    return true;
}

bool try_kuhn(const std::vector<std::vector<uint16_t>>& g, std::vector<uint16_t>& r_match,
              std::vector<uint8_t>& was, uint16_t from) {
    if (was[from]) return false;
    was[from] = 1u;
    for (uint16_t i = 0; i < g[from].size(); ++i) {
        uint16_t to = g[from][i];
        if (r_match[to] == 1234u || try_kuhn(g, r_match, was, r_match[to])) {
            r_match[to] = from;
            return true;
        }
    }
    return false;
}

void dfs(const std::vector<std::vector<uint16_t>>& g, std::vector<uint16_t>& min_cover,
         std::vector<uint8_t>& was, uint16_t from, int is_left) {
    was[from] = 1u;
    if (!is_left) min_cover.emplace_back(from);
    for (uint16_t i = 0; i < g[from].size(); ++i) {
        uint16_t to = g[from][i];
        if (!was[to]) {
            dfs(g, min_cover, was, to, is_left ^ 1);
        }
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
    uint16_t n, k;
    std::cin >> n >> k;
    std::vector<InputData> inputs;
    std::vector<std::vector<uint16_t>> points_w_k_ones(k + 1u);
    for (uint16_t i = 0; i < n; ++i) {
        inputs.emplace_back(InputData());
        for(uint16_t j = 0; j < k; ++j) {
            uint16_t b;
            std::cin >> b;
            inputs[i].in.push_back(b);
        }
        std::cin >> inputs[i].out;
        points_w_k_ones[inputs[i].get_num_ones()].push_back(i);
    }

    std::vector<std::vector<uint16_t>> wrong_hesse_graph(n);
    std::vector<uint16_t> l_verts;
    for (uint16_t i = 0; i <= k; ++i) {
        for (uint16_t j = 0; j < points_w_k_ones[i].size(); ++j) {
            uint16_t from = points_w_k_ones[i][j];
            for (uint16_t ii = i + 1; ii <= k; ++ii) {
                for (uint16_t jj = 0; jj < points_w_k_ones[ii].size(); ++jj) {
                    uint16_t to = points_w_k_ones[ii][jj];
                    if (inputs[from].out > inputs[to].out && check_order(inputs[from], inputs[to])) {
                        wrong_hesse_graph[from].emplace_back(to);
                        wrong_hesse_graph[to].emplace_back(from);
                        if (l_verts.empty() || l_verts.back() != from) l_verts.emplace_back(from);
                    }
                }
            }
        }
    }

    std::vector<uint16_t> r_match(n, 1234u);
    std::vector<uint8_t> was(n, 0u);
	for (uint16_t i = 0; i < l_verts.size(); ++i) {
		was.assign(n, 0u);
		try_kuhn(wrong_hesse_graph, r_match, was, l_verts[i]);
	}

    std::vector<std::vector<uint16_t>> match_graph(n);
    std::vector<uint8_t> in_match(n, 0u);
    for (uint16_t i = 0; i < l_verts.size(); ++i) {
        uint16_t from = l_verts[i];
        for (uint16_t j = 0; j < wrong_hesse_graph[from].size(); ++j) {
            uint16_t to = wrong_hesse_graph[from][j];
            if (r_match[to] == from) {
                match_graph[to].emplace_back(from);
                in_match[from] = 1u;
                in_match[to] = 1u;
            } else {
                match_graph[from].emplace_back(to);
            }
        }
    }

    std::vector<uint16_t> l_free;
    for (uint16_t i = 0; i < l_verts.size(); ++i) {
        if (!in_match[l_verts[i]]) l_free.emplace_back(l_verts[i]);
    }

    was.assign(n, 0u);
    std::vector<uint16_t> min_cover;
    for (uint16_t i = 0; i < l_free.size(); ++i) {
        dfs(match_graph, min_cover, was, l_free[i], 1);
    }
    for (uint16_t i = 0; i < l_verts.size(); ++i) {
        if (!was[l_verts[i]]) min_cover.emplace_back(l_verts[i]);
    }

    std::cout << min_cover.size() << "\n";
    for (uint16_t i = 0; i < min_cover.size(); ++i) {
        std::cout << min_cover[i] + 1 << " ";
    }
    return 0;
}
