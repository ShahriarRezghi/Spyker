#include "csv.h"

namespace Spyker
{
namespace Helper
{
CSV::CSV(const std::string &path, const std::string &delim) : stream(path), delim(delim) {}

bool CSV::_readline(std::vector<std::string> &row)
{
    row.clear();
    if (!std::getline(stream, line)) return false;
    if (line.empty()) return true;

    size_t prev = 0, index = 0;
    while (index != std::string::npos)
    {
        auto end = (index = line.find(delim, prev));
        if (index != std::string::npos) end -= prev;
        row.push_back(line.substr(prev, end));
        prev = index + 1;
    }
    return true;
}

bool CSV::readline(std::vector<std::string> &row)
{
    while (true)
    {
        if (!_readline(row)) return false;
        if (!line.empty()) return true;
    }
}

Size CSV::skip(Size lines)
{
    Size skiped = 0;
    for (; skiped < lines; ++skiped)
        if (!std::getline(stream, line)) break;
    return skiped;
}
}  // namespace Helper
}  // namespace Spyker
