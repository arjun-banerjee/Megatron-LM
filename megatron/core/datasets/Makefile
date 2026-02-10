CXXFLAGS += -O3 -Wall -shared -std=c++17 -fPIC -fdiagnostics-color
CPPFLAGS += $(shell /opt/conda/envs/megatron/bin/python -m pybind11 --includes)

LIBNAME = helpers_cpp
LIBEXT = $(shell /opt/conda/envs/megatron/bin/python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

OUT = $(LIBNAME)$(LIBEXT)
SRC = helpers.cpp

default: $(OUT)

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@
