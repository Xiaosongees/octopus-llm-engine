#include "octopus/server.hpp"
#include "json_io.hpp"
#include <chrono>
#include <httplib.h>
#include <iostream>

namespace octopus {
void serve(const InferenceEngine &engine, const std::string &host, int port, std::size_t threads) {
    using namespace detail;
    if (port < 0 || port > 65535 || threads < 1 || threads > 64)
        throw std::invalid_argument("Port must be 0-65535 and threads 1-64");
    httplib::Server server;
    server.new_task_queue = [threads] { return new httplib::ThreadPool(threads, threads, 64); };
    server.set_payload_max_length(1024 * 1024);
    server.set_read_timeout(5);
    server.set_write_timeout(5);
    server.set_keep_alive_max_count(20);
    auto start = std::chrono::steady_clock::now();
    auto error = [](httplib::Response &response, int status, const std::string &message) {
        response.status = status;
        response.set_content(Json{{"error", message}}.dump(), "application/json");
    };
    server.Get("/v1/health", [&](const httplib::Request &, httplib::Response &response) {
        response.set_content(
            Json{{"status", "ok"},
                 {"version", OCTOPUS_VERSION},
                 {"uptime_secs",
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()}}
                .dump(),
            "application/json");
    });
    server.Get("/v1/model/info", [&](const httplib::Request &, httplib::Response &response) {
        Json shapes = Json::object();
        for (const auto &name : engine.graph().input_names())
            shapes[name] = engine.graph().node(name).attributes.shape;
        response.set_content(Json{{"model_loaded", true},
                                  {"node_count", engine.graph().nodes().size()},
                                  {"backend", engine.backend_name()},
                                  {"inputs", shapes},
                                  {"outputs", engine.graph().output_names()}}
                                 .dump(),
                             "application/json");
    });
    server.Post("/v1/inference", [&](const httplib::Request &request, httplib::Response &response) {
        if (!request.get_header_value("Content-Type").starts_with("application/json")) {
            error(response, 415, "Content-Type must be application/json");
            return;
        }
        try {
            auto body = parse_request(request.body);
            auto inputs = decode_inputs(body);
            auto output = body.value("output_name", engine.graph().output_names().front());
            if (std::find(engine.graph().output_names().begin(), engine.graph().output_names().end(),
                          output) == engine.graph().output_names().end())
                throw std::invalid_argument("Unknown output: " + output);
            response.set_content(encode_result(engine.run(inputs), output).dump(), "application/json");
        } catch (const Json::exception &exception) {
            error(response, 400, std::string("Invalid inference JSON: ") + exception.what());
        } catch (const std::invalid_argument &exception) {
            error(response, 400, exception.what());
        } catch (const std::bad_alloc &) {
            error(response, 503, "Insufficient memory for inference");
        } catch (const std::exception &exception) {
            std::cerr << "Inference error: " << exception.what() << '\n';
            error(response, 500, "Inference failed");
        }
    });
    server.set_error_handler([&](const httplib::Request &, httplib::Response &response) {
        if (response.body.empty())
            error(response, response.status,
                  response.status == 413 ? "Request body exceeds 1 MiB" : "HTTP request failed");
    });
    auto bound_port = port;
    if (port == 0)
        bound_port = server.bind_to_any_port(host);
    else if (!server.bind_to_port(host, port))
        bound_port = -1;
    if (bound_port < 0)
        throw std::runtime_error("Cannot bind HTTP listener");
    std::cout << "Listening on http://" << host << ':' << bound_port << std::endl;
    if (!server.listen_after_bind())
        throw std::runtime_error("HTTP listener failed");
}
} // namespace octopus
