find_package(pybind11 REQUIRED)

pybind11_add_module(ebsdlib ${EbsdLibProj_SOURCE_DIR}/Source/Binding/py_ebsdlib.cpp)

target_link_libraries(ebsdlib
  PUBLIC
    EbsdLib
)

set_property(TARGET ebsdlib PROPERTY FOLDER "EbsdLibProj/Python")
