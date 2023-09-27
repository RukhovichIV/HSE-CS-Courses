#include <algorithm>
#include <iostream>
#include <limits>
#include <utility>
#include <optional>
#include <vector>

int main()
{
    std::cin.tie(0), std::cout.tie(0), std::ios_base::sync_with_stdio(0);
#define file
#ifdef file
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
#endif
    size_t gems_num, min_w, max_w;
    std::cin >> gems_num >> min_w >> max_w;
    std::vector<std::pair<size_t, size_t>> gems(gems_num);
    for (size_t i = 0; i < gems_num; ++i) {
        size_t weight, cost;
        std::cin >> weight >> cost;
        gems[i] = std::make_pair(weight, cost);
    }

    std::cout << gems[0].first << " " << gems[1].second << "\n";
    return 0;
}
