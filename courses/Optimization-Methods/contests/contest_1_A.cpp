#include <algorithm>
#include <iostream>
#include <limits>
#include <utility>
#include <optional>
#include <vector>
#include <deque>

class Thing {
public:
    uint32_t weight = 0, cost = 0, count = 0;
};

int main()
{
    std::cin.tie(0), std::cout.tie(0), std::ios_base::sync_with_stdio(0);
// #define file
#ifdef file
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif
    size_t types_num, backpack_size;
    std::cin >> types_num >> backpack_size;
    std::vector<Thing> things(types_num);
    for (size_t i = 0; i < types_num; ++i) {
        std::cin >> things[i].weight >> things[i].cost >> things[i].count;
    }

    std::vector<size_t> pack_old(backpack_size + 1u, 0u), pack_new(backpack_size + 1u, 0u);
    for (size_t i = 0u; i < types_num; ++i) {
        for (uint32_t j = 1u; j <= backpack_size; ++j) {
            pack_new[j] = std::max(pack_new[j - 1u], pack_old[j]);

            size_t le = 1u, ri = std::min(things[i].count, j / things[i].weight);
            while (le + 3u < ri) {
                size_t ml = le + (ri - le) / 3u, mr = ri - (ri - le) / 3u;
                size_t vle = pack_old[j - ml * things[i].weight] + ml * things[i].cost;
                size_t vri = pack_old[j - mr * things[i].weight] + mr * things[i].cost;
                if (vle < vri) {
                    le = ml;
                } else {
                    ri = mr;
                }
            }
            for (uint32_t cnt = le; cnt <= ri; ++cnt) {
                size_t new_val = pack_old[j - cnt * things[i].weight] + cnt * things[i].cost;
                if (new_val > pack_new[j]) {
                    pack_new[j] = new_val;
                }
            }

            // for (size_t cnt = 1; cnt <= things[i].count; ++cnt) {
            //     if (cnt * things[i].weight > j) {
            //         break;
            //     }
            //     size_t new_val = pack_old[j - cnt * things[i].weight] + cnt * things[i].cost;
            //     if (new_val > pack_new[j]) {
            //         pack_new[j] = new_val;
            //     }
            // }
        }
        std::swap(pack_new, pack_old);
    }

    std::cout << pack_old[backpack_size];

    // std::cout << "\n\n";
    // for (auto&& elem : pack_new) {
    //     std::cout << elem << " ";
    // }
    // std::cout << "\n";
    // for (auto&& elem : pack_old) {
    //     std::cout << elem << " ";
    // }

    return 0;
}
