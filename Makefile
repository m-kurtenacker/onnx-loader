.PHONY: all clean clean-all
all: a.out

clean:
	rm -f *.ll *.thorin.json a.out

clean-all: clean
	rm -rf plugin/build
	rm -rf pythorin/__pycache__
	rm -rf onnx_c/build onnx_c/install
	rm -rf venv

LOG_LEVEL=info

RUNTIME=${THORIN_RUNTIME_PATH}/artic/runtime.impala \
	${THORIN_RUNTIME_PATH}/artic/intrinsics_thorin.impala \
	${THORIN_RUNTIME_PATH}/artic/intrinsics_rv.impala \
	${THORIN_RUNTIME_PATH}/artic/intrinsics_math.impala \
	${THORIN_RUNTIME_PATH}/artic/intrinsics.impala

onnx_c/install:
	mkdir -p onnx_c/build
	cd onnx_c/build && \
		cmake .. -GNinja -DCMAKE_CXX_STANDARD=17 -DCMAKE_C_STANDARD=17 -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_INSTALL_PREFIX=../install && \
		ninja install

venv:
	python -m venv venv
	cd onnx_c && \
		../venv/bin/pip install -e .

plugin/build: onnx_c/install
	mkdir -p plugin/build
	cd plugin/build && cmake .. -DONNX_DIR=`pwd`/../../onnx_c/install/lib/cmake/ONNX

plugin/build/loader.so: plugin/load.cpp plugin/memory.cpp plugin/build
	@make -C plugin/build all

plugin/build/loader_runtime.so: plugin/load_runtime.cpp plugin/build
	@make -C plugin/build all

main.thorin.json: main.art read.art utils.art sequential.art mat.art
	artic \
		${RUNTIME} \
		$^ \
		--emit-json \
		--log-level ${LOG_LEVEL} \
		-o main

main-compiled.thorin.json: main.thorin.json plugin/build/loader.so
	anyopt \
		main.thorin.json \
		--pass cleanup \
		--pass lower2cff \
		--pass plugin_execute \
		--pass cleanup \
		--plugin plugin/build/loader.so \
		--keep-intern run_network \
		--emit-json \
		--log-level ${LOG_LEVEL} \
		-o main-compiled

network-tools.thorin.json: sequential.art mat.art
	artic \
		${RUNTIME} \
		$^ \
		--emit-json \
		--log-level ${LOG_LEVEL} \
		-o network-tools

network.thorin.json: network.py network-tools.thorin.json venv
	./venv/bin/python network.py

network-combined.thorin.json: network-tools.thorin.json network.thorin.json
	anyopt \
		$^ \
		--pass cleanup \
		--keep-intern run_network \
		--emit-json \
		--log-level ${LOG_LEVEL} \
		-o network-combined

network-compiled.thorin.json: network-combined.thorin.json plugin/build/loader.so
	anyopt \
		network-combined.thorin.json \
		--pass cleanup \
		--pass lower2cff \
		--pass plugin_execute \
		--pass cleanup \
		--plugin plugin/build/loader.so \
		--emit-json \
		--log-level ${LOG_LEVEL} \
		-o network-compiled

combined.ll: main-compiled.thorin.json network-compiled.thorin.json
	anyopt \
		$^ \
		--pass cleanup \
		--pass lower2cff \
		--pass flatten_tuples \
		--pass split_slots \
		--pass lift_builtins \
		--pass inliner \
		--pass hoist_enters \
		--pass dead_load_opt \
		--pass cleanup \
		--pass codegen_prepare \
		--emit-llvm \
		--log-level ${LOG_LEVEL} \
		-o combined

a.out: combined.ll utils/allocator.cpp utils/read.cpp plugin/build/loader_runtime.so
	clang++ -O3 -L${THORIN_RUNTIME_PATH}/../build/lib -lruntime -lm $^

#a.out: combined.ll utils/allocator.cpp utils/read.cpp
#	clang++ -o a.out -O3 -L${THORIN_RUNTIME_PATH}/../build/lib -lruntime -lm $^
#	#clang++ -Og -g -L${THORIN_RUNTIME_PATH}/../build/lib -lruntime -lm $^
