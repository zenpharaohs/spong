cmake_minimum_required(VERSION 3.16)
project(spong_core VERSION 0.0.1 LANGUAGES C)

option(SPONG_BUILD_SHARED "Build spong_core as a shared library" OFF)
option(SPONG_BUILD_C_TESTS "Build frontend-independent C tests" ON)

set(SPONG_LIBRARY_TYPE STATIC)
if(SPONG_BUILD_SHARED)
    set(SPONG_LIBRARY_TYPE SHARED)
endif()

add_library(spong_core ${SPONG_LIBRARY_TYPE}
    src/c/spong_resolution.c
    src/c/spong_exact_gmp.c
    src/c/spong_topology.c
    src/c/spong_geometry.c
    src/c/spong_local_gmp.c
    src/c/spong_continue.c
    src/c/spong_gauss2.c
    src/c/spong_potential.c
    src/c/spong_jet.c
    src/c/spong_arrival.c
)
add_library(spong::core ALIAS spong_core)
target_include_directories(spong_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(spong_core PUBLIC c_std_99)
find_path(GMP_INCLUDE_DIR gmp.h)
find_library(GMP_LIBRARY gmp)
if(NOT GMP_INCLUDE_DIR OR NOT GMP_LIBRARY)
    message(FATAL_ERROR
        "GMP is required for certified exact arithmetic; set "
        "GMP_INCLUDE_DIR and GMP_LIBRARY for cross builds")
endif()
target_include_directories(spong_core PRIVATE ${GMP_INCLUDE_DIR})
target_link_libraries(spong_core PUBLIC ${GMP_LIBRARY})
set_target_properties(spong_core PROPERTIES
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    PUBLIC_HEADER
        "include/spong/spong_resolution.h;include/spong/spong_exact.h;include/spong/spong_topology.h;include/spong/spong_geometry.h;include/spong/spong_local.h;include/spong/spong_continue.h;include/spong/spong_gauss2.h;include/spong/spong_potential.h;include/spong/spong_jet.h;include/spong/spong_arrival.h"
)
if(SPONG_BUILD_SHARED)
    target_compile_definitions(spong_core
        PUBLIC SPONG_BUILD_SHARED
        PRIVATE SPONG_BUILDING_LIBRARY
    )
endif()

if(NOT MSVC)
    target_link_libraries(spong_core PUBLIC m)
endif()

if(SPONG_BUILD_C_TESTS)
    enable_testing()
    add_executable(test_spong_resolution tests/c/test_spong_resolution.c)
    target_link_libraries(test_spong_resolution PRIVATE spong::core)
    add_test(NAME spong_resolution_c_abi COMMAND test_spong_resolution)
    add_executable(test_spong_exact tests/c/test_spong_exact.c)
    target_link_libraries(test_spong_exact PRIVATE spong::core)
    add_test(NAME spong_exact_c_abi COMMAND test_spong_exact)
    add_executable(test_spong_topology tests/c/test_spong_topology.c)
    target_link_libraries(test_spong_topology PRIVATE spong::core)
    add_test(NAME spong_topology_c_abi COMMAND test_spong_topology)
    add_executable(test_spong_geometry tests/c/test_spong_geometry.c)
    target_link_libraries(test_spong_geometry PRIVATE spong::core)
    add_test(NAME spong_geometry_c_abi COMMAND test_spong_geometry)
    add_executable(test_spong_local tests/c/test_spong_local.c)
    target_link_libraries(test_spong_local PRIVATE spong::core)
    add_test(NAME spong_local_c_abi COMMAND test_spong_local)
    add_executable(test_spong_gauss2 tests/c/test_spong_gauss2.c)
    target_link_libraries(test_spong_gauss2 PRIVATE spong::core m)
    add_test(NAME spong_gauss2_c_abi COMMAND test_spong_gauss2)
endif()

include(GNUInstallDirs)
install(TARGETS spong_core
    EXPORT spongCoreTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/spong
)
install(EXPORT spongCoreTargets
    NAMESPACE spong::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/spong
)
