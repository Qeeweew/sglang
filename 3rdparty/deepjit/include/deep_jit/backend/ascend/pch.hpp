#pragma once

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <deep_jit/cache/compile_lock.hpp>
#include <deep_jit/cache/disk.hpp>
#include <deep_jit/runtime/config.hpp>
#include <deep_jit/utils/command.hpp>
#include <deep_jit/utils/env.hpp>
#include <deep_jit/utils/filesystem.hpp>
#include <deep_jit/utils/hash.hpp>
#include <deep_jit/utils/str.hpp>

namespace deep_jit::ascend {

// Bisheng's dav-2201 driver parses the SDK once for each actor. Obtain its
// actual cc1 commands with an empty entry, then reuse an actor-specific SDK
// PCH. Component -D flags are applied to the consumer, never baked into PCH.
// These project flags must not change SDK header conditional compilation.
class SdkPch {
    using Command = std::vector<std::string>;
    static constexpr const char* kHeader =
        "#include <kernel_operator.h>\n#include <c_api/asc_simd.h>\n";
    static constexpr const char* kProbe =
        "extern \"C\" __global__ __vector__ void deepjit_compile_probe() {}\n";

    static size_t option(const Command& command, const std::string& name) {
        const auto position = std::ranges::find(command, name);
        DJ_HOST_ASSERT(position != command.end() and position + 1 != command.end(),
                       "missing Bisheng command option: {}", name);
        return static_cast<size_t>(position - command.begin()) + 1;
    }

    static Command split(const std::string& line) {
        // Clang's verbose driver quotes whole arguments with std::quoted's
        // double-quote/backslash convention. This is not a shell evaluator.
        std::istringstream input(line);
        Command result;
        std::string token;
        while (input >> std::quoted(token)) result.push_back(token);
        DJ_HOST_ASSERT(input.eof(), "invalid Bisheng driver command: {}", line);
        return result;
    }

    static std::vector<Command> commands(const std::string& log) {
        std::vector<Command> result;
        std::istringstream input(log);
        std::string line;
        while (std::getline(input, line)) {
            if (line.find(" -cc1 ") == std::string::npos or
                line.find(" -triple ") == std::string::npos) continue;
            auto command = split(line);
            DJ_HOST_ASSERT(command.size() > 2 and command[1] == "-cc1",
                           "unexpected Bisheng frontend command");
            std::erase(command, "-v");
            result.push_back(std::move(command));
        }
        DJ_HOST_ASSERT(result.size() == 2, "expected Cube and Vector compiler commands, got {}", result.size());
        DJ_HOST_ASSERT(result[0][option(result[0], "-target-cpu")] == "dav-c220-cube" and
                       result[1][option(result[1], "-target-cpu")] == "dav-c220-vec",
                       "SDK PCH supports the dav-2201 Cube/Vector compiler plan");
        return result;
    }

    static std::string serialize(const std::vector<Command>& commands) {
        std::ostringstream output;
        for (const auto& command : commands) {
            for (const auto& argument : command) output << std::quoted(argument) << ' ';
            output << '\n';
        }
        return output.str();
    }

    static std::vector<Command> deserialize(const std::string& data) {
        std::vector<Command> result;
        std::istringstream input(data);
        std::string line;
        while (std::getline(input, line)) result.push_back(split(line));
        DJ_HOST_ASSERT(result.size() == 2, "invalid cached SDK PCH plan");
        return result;
    }

    static std::filesystem::path prepare(const Command& base, const std::string& identity,
                                         const Env& env, bool print_command) {
        auto cache = DiskCache::from_env(env);
        // PCH stores absolute paths to its inputs. Publish those tiny inputs
        // first, so the later atomic PCH rename cannot invalidate them.
        const auto input_key = hash::FNV1a().update(kHeader).update(kProbe).get_hex_digest();
        auto inputs = cache.entry("ascend_pch_source", input_key);
        if (not inputs.hit) {
            write_file_sync(inputs.path / "preamble.asc", kHeader);
            write_file_sync(inputs.path / "kernel.asc", kProbe);
            inputs.commit();
        }

        const auto key = hash::FNV1a().update("sdk-pch-v1").update(identity)
            .update(str::shell_join(base)).update(inputs.path.string()).get_hex_digest();
        size_t root;
        {
            auto entry = cache.entry("ascend_sdk_pch", key);
            if (entry.hit) return entry.path;
            root = entry.root_index;
        }
        CompileLock lock(cache.paths[root], key);
        auto entry = cache.entry("ascend_sdk_pch", key, root);
        if (entry.hit) return entry.path;

        auto probe = base;
        probe.insert(probe.end(), {"-v", (inputs.path / "kernel.asc").string(), "-o",
                                  (entry.path / "probe.rel.o").string()});
        const auto log = call_external_command(str::shell_join(probe), print_command);
        auto plan = commands(log);
        write_file_sync(entry.path / "driver.log", log);
        write_file_sync(entry.path / "plan.commands", serialize(plan));
        for (size_t actor = 0; actor < plan.size(); ++actor) {
            auto command = plan[actor];
            const auto action = std::ranges::find(command, "-emit-obj");
            DJ_HOST_ASSERT(action != command.end(), "Bisheng frontend did not emit an object");
            *action = "-emit-pch";
            command[option(command, "-o")] = (entry.path / (std::to_string(actor) + ".pch")).string();
            command.back() = (inputs.path / "preamble.asc").string();
            call_external_command(str::shell_join(command), print_command);
        }
        safe_remove_all(entry.path / "probe.rel.o");
        return entry.commit();
    }

public:
    // Returns the exact commands used, for the ordinary per-kernel meta.json.
    static std::string compile(const Command& base, const Command& defines,
                               const std::string& identity, const Env& env,
                               const std::filesystem::path& source,
                               const std::filesystem::path& output,
                               const std::filesystem::path& linker, bool print_command) {
        const auto directory = prepare(base, identity, env, print_command);
        const auto plan = deserialize(read(directory / "plan.commands"));
        std::string executed;
        Command link{linker.string(), "-m", "aicorelinux", "-Ttext", "0"};
        for (size_t actor = 0; actor < plan.size(); ++actor) {
            auto command = plan[actor];
            const auto object = output.parent_path() / ("kernel." + std::to_string(actor) + ".o");
            command[option(command, "-o")] = object.string();
            command.back() = source.string();
            command.insert(command.end(), {"-include-pch", (directory / (std::to_string(actor) + ".pch")).string()});
            command.insert(command.end(), defines.begin(), defines.end());
            const auto text = str::shell_join(command);
            call_external_command(text, print_command);
            if (not executed.empty()) executed += " && ";
            executed += text;
            link.push_back(object.string());
        }
        link.insert(link.end(), {"-o", output.string(), "-r", "--allow-multiple-definition"});
        const auto text = str::shell_join(link);
        call_external_command(text, print_command);
        for (size_t actor = 0; actor < plan.size(); ++actor)
            safe_remove_all(output.parent_path() / ("kernel." + std::to_string(actor) + ".o"));
        return executed + " && " + text;
    }
};

} // namespace deep_jit::ascend
