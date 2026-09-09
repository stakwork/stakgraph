// @ast node: Class "Config"
// @ast node: Function "getVersion"
// @ast node: Class "Wrapper"
// @ast node: Function "getValue"
#pragma once
#include <string>

namespace app {
class Config {
public:
    std::string getVersion() { return version; }
private:
    std::string version = "1.0";
};
}

template<typename T>
class Wrapper {
public:
    T getValue() { return value; }
private:
    T value;
};
