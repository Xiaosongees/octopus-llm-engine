#pragma once
#include "octopus/inference.hpp"
namespace octopus {
void serve(const InferenceEngine &engine, const std::string &host = "127.0.0.1", int port = 8080,
           std::size_t threads = 4);
}
