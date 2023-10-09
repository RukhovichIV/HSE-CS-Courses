#include <algorithm>
#include <iostream>
#include <limits>
#include <utility>
#include <optional>
#include <vector>
#include <deque>

int main()
{
    std::cin.tie(0), std::cout.tie(0), std::ios_base::sync_with_stdio(0);
// #define file
#ifdef file
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif
    uint64_t types_num, backpack_size;
    std::cin >> types_num >> backpack_size;
    std::vector<std::pair<uint64_t, uint64_t>> things;
    for (size_t i = 0; i < types_num; ++i) {
        uint64_t weight, cost, count;
        std::cin >> weight >> cost >> count;

        uint64_t j = 1u;
        while (j <= count) {
            if (j * weight <= backpack_size) {
                things.emplace_back(j * weight, j * cost);
            }
            count -= j;
            j *= 2u;
        }
        if (count > 0 && count * weight <= backpack_size) {
            things.emplace_back(count * weight, count * cost);
        }
    }

    std::vector<uint64_t> pack(backpack_size + 1u, 0u);
    for (size_t i = 0u; i < things.size(); ++i) {
        for (uint64_t j = backpack_size; j >= things[i].first; --j) {
            pack[j] = std::max(pack[j], pack[j - things[i].first] + things[i].second);
        }
    }

    std::cout << pack[backpack_size];

    return 0;
}
