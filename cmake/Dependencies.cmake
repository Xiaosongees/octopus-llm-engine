# Only the JSON codec and optional HTTP transport use third-party headers.
# Dependencies are downloaded into the build tree, never into the source tree.
if(OCTOPUS_DEPENDENCY_DIR)
    set(_octopus_deps "${OCTOPUS_DEPENDENCY_DIR}")
else()
    set(_octopus_deps "${CMAKE_BINARY_DIR}/_deps/include")
endif()

function(octopus_header relative_path url sha256)
    set(destination "${_octopus_deps}/${relative_path}")
    if(NOT EXISTS "${destination}")
        if(OCTOPUS_DEPENDENCY_DIR)
            message(FATAL_ERROR "Missing offline dependency: ${destination}")
        endif()
        get_filename_component(directory "${destination}" DIRECTORY)
        file(MAKE_DIRECTORY "${directory}")
        message(STATUS "Downloading ${relative_path}")
        file(DOWNLOAD "${url}" "${destination}" EXPECTED_HASH "SHA256=${sha256}"
            TLS_VERIFY ON TIMEOUT 120 STATUS status)
        list(GET status 0 code)
        if(NOT code EQUAL 0)
            message(FATAL_ERROR "Dependency download failed: ${status}. Set OCTOPUS_DEPENDENCY_DIR for an offline build.")
        endif()
    endif()
    file(SHA256 "${destination}" actual_sha256)
    if(NOT actual_sha256 STREQUAL sha256)
        message(FATAL_ERROR "Dependency checksum mismatch: ${destination}")
    endif()
endfunction()

octopus_header(nlohmann/json.hpp
    https://raw.githubusercontent.com/nlohmann/json/v3.12.0/single_include/nlohmann/json.hpp
    aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63)
add_library(octopus_json INTERFACE)
target_include_directories(octopus_json SYSTEM INTERFACE "${_octopus_deps}")
if(OCTOPUS_BUILD_SERVER)
    octopus_header(httplib.h
        https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.63.0/httplib.h
        9021001a719768893a8a6e4051c56107690e8b8e49b41e5f903534fa19ae9afb)
    add_library(octopus_http INTERFACE)
    target_include_directories(octopus_http SYSTEM INTERFACE "${_octopus_deps}")
    target_link_libraries(octopus_http INTERFACE Threads::Threads)
    if(WIN32)
        target_link_libraries(octopus_http INTERFACE ws2_32 crypt32)
    endif()
endif()
