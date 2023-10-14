#include <algorithm>
#include <iostream>
#include <limits>
#include <utility>
#include <optional>
#include <vector>
#include <deque>

class Combination {
public:
    size_t weight = 0, cost = 0;
    std::vector<size_t> numbers;
};

bool compare_combinations(const Combination& lhs, const Combination& rhs) {
    return lhs.weight < rhs.weight;
}

void push_front(std::deque<Combination>& deque, Combination& comb) {
    while (!deque.empty() && deque.front().cost < comb.cost) {
        deque.pop_front();
    }
    deque.push_front(comb);
}

void pop_back(std::deque<Combination>& deque, const size_t& max_weight) {
    while (!deque.empty() && deque.back().weight > max_weight) {
        deque.pop_back();
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
    size_t gems_num, min_w, max_w;
    std::cin >> gems_num >> min_w >> max_w;
    std::vector<std::pair<size_t, size_t>> gems(gems_num);
    for (size_t i = 0; i < gems_num; ++i) {
        size_t weight, cost;
        std::cin >> weight >> cost;
        gems[i] = std::make_pair(weight, cost);
    }
    int proc_fi = gems_num / 2, proc_se = static_cast<int>(gems_num - proc_fi);
    std::vector<Combination> combs_le(1 << proc_fi), combs_ri(1 << proc_se);

    for (int i = 0; i < 1 << proc_fi; ++i) {
        for (int j = 0; j < proc_fi; ++j) {
            if (i & (1 << j)) {
                combs_le[i].weight += gems[j].first;
                combs_le[i].cost += gems[j].second;
                combs_le[i].numbers.push_back(j + 1);
            }
        }
    }
    for (int i = 0; i < 1 << proc_se; ++i) {
        for (int j = 0; j < proc_se; ++j) {
            if (i & (1 << j)) {
                combs_ri[i].weight += gems[proc_fi + j].first;
                combs_ri[i].cost += gems[proc_fi + j].second;
                combs_ri[i].numbers.push_back(proc_fi + j + 1);
            }
        }
    }
    std::sort(combs_le.begin(), combs_le.end(), compare_combinations);
    std::sort(combs_ri.begin(), combs_ri.end(), compare_combinations);

    size_t se_le = combs_ri.size() - 1;
    std::deque<Combination> combs_proc;
    Combination ans_default = Combination();
    ans_default.weight = -1;
    Combination ans_le = Combination(), ans_ri = ans_default;
    for (size_t i = 0; i < combs_le.size(); ++i) {
        if (combs_le[i].weight > max_w) {
            break;
        }
        while (combs_le[i].weight + combs_ri[se_le].weight >= min_w) {
            if (se_le == 0) {
                if (combs_proc.empty() || combs_proc.front().cost != 0) {
                    push_front(combs_proc, combs_ri[se_le]);
                }
                break;
            }
            push_front(combs_proc, combs_ri[se_le]);
            --se_le;
        }
        pop_back(combs_proc, max_w - combs_le[i].weight);
        if (!combs_proc.empty() && (ans_ri.weight == ans_default.weight ||
            (combs_le[i].cost + combs_proc.back().cost > ans_le.cost + ans_ri.cost))) {
            ans_le = combs_le[i];
            ans_ri = combs_proc.back();

            // if (ans_ri.weight == static_cast<size_t>(-1)) {
            //     std::cout << "No answer\n";
            // }
            // std::cout << ans_le.weight + ans_ri.weight << " ";
            // std::cout << ans_le.cost + ans_ri.cost << " | ";
            // for (auto&& it : ans_le.numbers) {
            //     std::cout << it << " ";
            // }
            // for (auto&& it : ans_ri.numbers) {
            //     std::cout << it << " ";
            // }
            // std::cout << "\n\n";
        }
    }

    size_t ans_size = ans_le.numbers.size() + ans_ri.numbers.size();
    std::cout << ans_size;
    if (ans_size) {
        std::cout << "\n";
        for (auto&& it : ans_le.numbers) {
            std::cout << it << " ";
        }
        for (auto&& it : ans_ri.numbers) {
            std::cout << it << " ";
        }
    }

    // std::cout << "\n\nList Combinations:\n";
    // for (size_t i = 0; i < combs_le.size(); ++i) {
    //     std::cout << combs_le[i].weight <<  " " << combs_le[i].cost << " | ";
    //     for (size_t jj = 0; jj < combs_le[i].numbers.size(); ++jj) {
    //         std::cout << combs_le[i].numbers[jj] << " ";
    //     }
    //     std::cout << "\n";
    // }
    // std::cout << "\n";
    // for (size_t i = 0; i < combs_ri.size(); ++i) {
    //     std::cout << combs_ri[i].weight <<  " " << combs_ri[i].cost << " | ";
    //     for (size_t jj = 0; jj < combs_ri[i].numbers.size(); ++jj) {
    //         std::cout << combs_ri[i].numbers[jj] << " ";
    //     }
    //     std::cout << "\n";
    // }
    return 0;
}
