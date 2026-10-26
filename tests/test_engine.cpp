#include "octopus/model_io.hpp"
#include <future>
#include <iostream>
#include <limits>

namespace {
using namespace octopus;
void check(bool condition, const std::string &message = "Check failed") {
    if (!condition)
        throw std::runtime_error(message);
}
void close(const TensorF32 &tensor, const std::vector<float> &expected, float tolerance = 1e-5f) {
    check(tensor.size() == expected.size(), "Element count differs");
    for (std::size_t i = 0; i < expected.size(); ++i)
        check(std::isfinite(tensor.values()[i]) && std::abs(tensor.values()[i] - expected[i]) <= tolerance,
              "Numerical mismatch at element " + std::to_string(i));
}
template <typename Function> void rejects(Function function) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception &) {
        rejected = true;
    }
    check(rejected, "Expected an exception");
}
using Test = std::pair<std::string, std::function<void()>>;
} // namespace
int main() {
    using namespace octopus;
    std::vector<Test> tests{
        {"tensor shape validation",
         [] {
             rejects([] { TensorF32({1, 2}, {3}); });
             rejects([] { TensorF32::zeros({0}); });
             rejects([] { element_count({std::numeric_limits<std::size_t>::max(), 2}); });
             TensorF32 scalar({2}, {});
             check(scalar.rank() == 0);
             close(scalar.add_scalar(3), {5});
         }},
        {"reshape shares immutable storage",
         [] {
             TensorF32 x({1, 2, 3, 4}, {2, 2});
             auto y = x.reshape({4});
             check(x.values().data() == y.values().data());
             rejects([&] { x.reshape({3}); });
             std::vector<std::size_t> index{1, 1};
             check(x.at(index) == 4);
             index[0] = 2;
             rejects([&] { x.at(index); });
         }},
        {"float64 tensor arithmetic",
         [] {
             TensorF64 x({1.0, 2.0}, {1, 2});
             TensorF64 y({3.0, 4.0}, {2, 1});
             check(std::abs(x.matmul(y).values()[0] - 11) < 1e-12);
         }},
        {"broadcast arithmetic",
         [] {
             TensorF32 x({1, 2, 3, 4, 5, 6}, {2, 3}), bias({10, 20, 30}, {3});
             close(x.add(bias), {11, 22, 33, 14, 25, 36});
             close(x.sub(TensorF32({1}, {})), {0, 1, 2, 3, 4, 5});
             close(x.mul(TensorF32({2, 3}, {2, 1})), {2, 4, 6, 12, 15, 18});
             rejects([&] { x.add(TensorF32::ones({2})); });
         }},
        {"broadcast batched matmul",
         [] {
             TensorF32 x({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});
             TensorF32 y({1, 0, 0, 2}, {2, 2});
             close(x.matmul(y), {1, 4, 3, 8, 5, 12, 7, 16});
             auto result = x.reshape({2, 1, 2, 2}).matmul(y.reshape({1, 1, 2, 2}));
             check(result.shape() == Shape({2, 1, 2, 2}));
             rejects([&] { x.matmul(TensorF32::ones({3, 2})); });
         }},
        {"transpose and permutation",
         [] {
             TensorF32 x({1, 2, 3, 4, 5, 6}, {2, 3});
             close(x.transpose(), {1, 4, 2, 5, 3, 6});
             close(x.reshape({1, 2, 3}).permute({2, 0, 1}), {1, 4, 2, 5, 3, 6});
             rejects([&] { x.permute({0, 0}); });
         }},
        {"stable softmax on multiple axes",
         [] {
             close(TensorF32({1000, 1000}, {2}).softmax(), {0.5f, 0.5f});
             auto x = TensorF32({0, 0, 0, 0, 0, 0}, {2, 3});
             close(x.softmax(), {1.f / 3, 1.f / 3, 1.f / 3, 1.f / 3, 1.f / 3, 1.f / 3});
             close(x.softmax(0), {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f});
             rejects([&] { x.softmax(-3); });
             rejects([] { TensorF32({-INFINITY, -INFINITY}, {2}).softmax(); });
             rejects([] { TensorF32({NAN, 1}, {2}).softmax(); });
         }},
        {"layer norm and reduction",
         [] {
             auto x = TensorF32({1, 3, 2, 4}, {2, 2});
             close(x.layer_norm(1e-8f), {-1, 1, -1, 1});
             close(x.sum_axis(0), {3, 7});
             close(x.sum_axis(1), {4, 6});
             close(x.layer_norm(1e-8f, {2, 2}).sum_axis(0).sum_axis(0), {0});
             rejects([&] { x.layer_norm(-1); });
             rejects([&] { x.layer_norm(1e-5f, {3}); });
         }},
        {"activation semantics",
         [] {
             TensorF32 x({-1, 0, 1}, {3});
             close(x.relu(), {0, 0, 1});
             close(x.gelu(), {-0.158808f, 0, 0.841192f}, 1e-5f);
             close(x.sigmoid(), {0.268941f, 0.5f, 0.731059f});
             close(x.tanh(), {-0.761594f, 0, 0.761594f});
         }},
        {"conv2d numerical output and padding",
         [] {
             TensorF32 x({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 3, 3});
             close(conv2d(x, TensorF32::ones({1, 1, 2, 2})), {12, 16, 24, 28});
             close(conv2d(x, TensorF32::ones({1, 1, 3, 3}), nullptr, {2, 2}, {1, 1}), {12, 16, 24, 28});
             rejects([&] { conv2d(x, TensorF32::ones({1, 2, 2, 2})); });
             rejects([&] { conv2d(x, TensorF32::ones({1, 1, 2, 2}), nullptr, {0, 1}); });
         }},
        {"embedding performs real lookup",
         [] {
             TensorF32 table({1, 2, 3, 4, 5, 6}, {3, 2});
             close(embedding(TensorF32({2, 0}, {1, 2}), table), {5, 6, 1, 2});
             rejects([&] { embedding(TensorF32({0.5f}, {1}), table); });
             rejects([&] { embedding(TensorF32({3}, {1}), table); });
             rejects([&] { embedding(TensorF32({INFINITY}, {1}), table); });
         }},
        {"attention masks and causal attention",
         [] {
             auto q = TensorF32::zeros({1, 1, 2, 1});
             TensorF32 v({1, 3}, {1, 1, 2, 1});
             close(scaled_dot_product_attention(q, q, v), {2, 2});
             close(scaled_dot_product_attention(q, q, v, nullptr, true), {1, 2});
             TensorF32 mask({0, -INFINITY, 0, -INFINITY}, {2, 2});
             close(scaled_dot_product_attention(q, q, v, &mask), {1, 1});
             rejects([&] { scaled_dot_product_attention(q, v.reshape({1, 1, 1, 2}), v); });
             close(scaled_dot_product_attention(TensorF32({1, 2}, {1, 1, 2, 1}),
                                                TensorF32({3, 4}, {1, 1, 2, 1}), v),
                   {2.462117f, 2.761594f});
         }},
        {"multi-head projections are split and concatenated correctly",
         [] {
             GraphBuilder b;
             b.input("q", {1, 2, 2});
             b.input("k", {1, 2, 2});
             b.input("v", {1, 2, 2});
             Attributes a;
             a.heads = 2;
             b.add_node({"attention", Op::Attention, {"q", "k", "v"}, {}, a});
             b.output("y", "attention");
             InferenceEngine e(b.build());
             auto result = e.run({{"q", TensorF32({1, 0, 0, 1}, {1, 2, 2})},
                                  {"k", TensorF32({1, 0, 0, 1}, {1, 2, 2})},
                                  {"v", TensorF32({1, 2, 3, 4}, {1, 2, 2})}});
             close(result.outputs.at("y"), {1.537883f, 3.0f, 2.0f, 3.462117f});
         }},
        {"graph repeated edges and unordered nodes",
         [] {
             Graph graph({{"output", Op::Output, {"twice"}, {}, {}},
                          {"twice", Op::Add, {"x", "x"}, {}, {}},
                          {"x", Op::Input, {}, {}, Attributes{Shape{2}}}});
             InferenceEngine engine(std::move(graph));
             close(engine.infer("x", TensorF32({2, 3}, {2})), {4, 6});
         }},
        {"graph rejects cycles, missing edges, and duplicate names",
         [] {
             rejects([] {
                 Graph({{"x", Op::Input, {}, {}, Attributes{Shape{1}}},
                        {"a", Op::Relu, {"b"}, {}, {}},
                        {"b", Op::Relu, {"a"}, {}, {}},
                        {"y", Op::Output, {"a"}, {}, {}}});
             });
             rejects([] {
                 GraphBuilder b;
                 b.input("x", {1});
                 b.output("y", "missing");
                 b.build();
             });
             rejects([] {
                 GraphBuilder b;
                 b.input("x", {1});
                 b.output("x", "x");
                 b.build();
             });
             rejects([] { GraphBuilder{}.build(); });
         }},
        {"dense batch bias and parameter validation",
         [] {
             GraphBuilder b;
             b.input("x", {2, 2});
             b.dense("fc", "x");
             b.output("y", "fc");
             rejects([&] { InferenceEngine(b.build()); });
             InferenceEngine engine(b.build(), {{"fc.weight", TensorF32({1, 0, 0, 1}, {2, 2})},
                                                {"fc.bias", TensorF32({1, 2}, {2})}});
             close(engine.infer("x", TensorF32({3, 4, 5, 6}, {2, 2})), {4, 6, 6, 8});
             rejects([&] { engine.infer("x", TensorF32::zeros({1, 2})); });
             rejects([&] { engine.run({}); });
             rejects([&] { InferenceEngine(b.build(), {}, BackendType::Cuda); });
         }},
        {"model inputs and outputs do not leak between runs",
         [] {
             GraphBuilder b;
             b.input("x", {1});
             b.input("z", {1});
             b.binary("sum", Op::Add, "x", "z");
             b.output("first", "sum");
             b.output("second", "x");
             InferenceEngine engine(b.build());
             auto first = engine.run({{"x", TensorF32({2}, {1})}, {"z", TensorF32({3}, {1})}});
             close(first.outputs.at("first"), {5});
             close(first.outputs.at("second"), {2});
             rejects([&] { engine.run({{"x", TensorF32({9}, {1})}}); });
             close(first.outputs.at("first"), {5});
             check(first.stats.nodes_executed == 3);
         }},
        {"concurrent runs use separate scratch storage",
         [] {
             auto engine = make_sample_engine();
             std::vector<std::future<void>> jobs;
             for (int i = 1; i <= 24; ++i)
                 jobs.push_back(std::async(std::launch::async, [&, i] {
                     auto x = static_cast<float>(i);
                     close(engine.infer("input", TensorF32({x, x, x, x}, {1, 4})),
                           {x + 0.5f, x - 0.5f, 2 * x + 1});
                 }));
             for (auto &job : jobs)
                 job.get();
         }},
        {"layer norm learned affine parameters broadcast",
         [] {
             GraphBuilder b;
             b.input("x", {2, 2});
             b.layer_norm("ln", "x", {2});
             b.output("y", "ln");
             InferenceEngine e(b.build(),
                               {{"ln.weight", TensorF32({2, 3}, {2})}, {"ln.bias", TensorF32({1, 2}, {2})}});
             close(e.infer("x", TensorF32({1, 3, 2, 4}, {2, 2})), {-1, 5, -1, 5}, 2e-5f);
         }},
        {"transformer computes nontrivial deterministic outputs",
         [] {
             TransformerConfig c;
             c.d_model = 4;
             c.n_heads = 2;
             c.n_layers = 1;
             c.d_ff = 8;
             c.vocab_size = 8;
             c.max_seq_len = 4;
             auto a = TransformerEncoder(c).build(2, 3, 7);
             auto b = TransformerEncoder(c).build(2, 3, 7);
             check(a.parameters.at("token_embedding.weight").values() ==
                   b.parameters.at("token_embedding.weight").values());
             auto stats = model_stats(a.parameters);
             check(stats.total_parameters > 0 && stats.weight_bytes == stats.total_parameters * 4);
             InferenceEngine engine(std::move(a.graph), std::move(a.parameters));
             auto output = engine.infer("tokens", TensorF32({1, 2, 3, 3, 2, 1}, {2, 3}));
             check(output.shape() == Shape({2, 3, 4}));
             for (auto value : output.values())
                 check(std::isfinite(value));
             check(std::abs(output.values()[0] - output.values()[1]) > 1e-4f);
             rejects([&] { TransformerEncoder(c).build(1, 5); });
             c.n_heads = 0;
             rejects([&] { TransformerEncoder invalid(c); });
         }},
        {"causal transformer does not read future tokens",
         [] {
             TransformerConfig c;
             c.d_model = 4;
             c.n_heads = 2;
             c.n_layers = 1;
             c.d_ff = 8;
             c.vocab_size = 8;
             c.causal = true;
             auto m = TransformerEncoder(c).build(1, 3, 42);
             InferenceEngine engine(std::move(m.graph), std::move(m.parameters));
             auto a = engine.infer("tokens", TensorF32({1, 2, 3}, {1, 3}));
             auto b = engine.infer("tokens", TensorF32({1, 6, 7}, {1, 3}));
             for (std::size_t i = 0; i < 4; ++i)
                 check(std::abs(a.values()[i] - b.values()[i]) < 1e-6f);
         }},
        {"reshape inference and invalid dimensions", [] {
             GraphBuilder b;
             b.input("x", {2, 3});
             Attributes a;
             a.target_shape = {3, -1};
             b.unary("r", Op::Reshape, "x", a);
             b.output("y", "r");
             auto result = InferenceEngine(b.build()).infer("x", TensorF32({1, 2, 3, 4, 5, 6}, {2, 3}));
             check(result.shape() == Shape({3, 2}));
             GraphBuilder invalid;
             invalid.input("x", {2, 3});
             a.target_shape = {-1, -1};
             invalid.unary("r", Op::Reshape, "x", a);
             invalid.output("y", "r");
             rejects([&] { InferenceEngine(invalid.build()).infer("x", TensorF32::ones({2, 3})); });
         }}};
    std::size_t failed = 0;
    for (const auto &[name, function] : tests) {
        try {
            function();
            std::cout << "PASS " << name << '\n';
        } catch (const std::exception &error) {
            ++failed;
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        }
    }
    std::cout << tests.size() - failed << '/' << tests.size() << " engine tests passed\n";
    return failed ? 1 : 0;
}
