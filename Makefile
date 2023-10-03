.PHONY: all clean clean-all
all: a.out

clean:
	rm -f *.ll *.thorin.json a.out test minimized

clean-all: clean
	@make -C plugin/build clean

LOG_LEVEL=info

RUNTIME=${THORIN_RUNTIME_PATH}/artic/runtime.impala \
	${THORIN_RUNTIME_PATH}/artic/intrinsics_thorin.impala \
	${THORIN_RUNTIME_PATH}/artic/intrinsics_rv.impala \
	${THORIN_RUNTIME_PATH}/artic/intrinsics_math.impala \
	${THORIN_RUNTIME_PATH}/artic/intrinsics.impala

.PHONY: onnx
onnx:
	mkdir -p onnx/build
	cd onnx/build && \
		cmake .. -GNinja -DCMAKE_CXX_STANDARD=17 -DCMAKE_C_STANDARD=17 -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_INSTALL_PREFIX=../install && \
		ninja install

plugin/build/loader.so: plugin/load.cpp plugin/memory.cpp
	@make -C plugin/build all

main.thorin.json: main.art read.art utils.art
	artic \
		${RUNTIME} \
		$^ \
		--emit-json \
		--log-level ${LOG_LEVEL} \
		-o main

network.thorin.json: network.art sequential.art mat.art
	artic \
		${RUNTIME} \
		$^ \
		--emit-json \
		--log-level ${LOG_LEVEL} \
		-o network

network-compiled.thorin.json: network.thorin.json plugin/build/loader.so
	anyopt \
		network.thorin.json \
		--pass cleanup \
		--pass lower2cff \
		--pass plugin_execute \
		--pass cleanup \
		--plugin plugin/build/loader.so \
		--emit-json \
		--log-level ${LOG_LEVEL} \
		-o network-compiled

combined.ll: main.thorin.json network-compiled.thorin.json plugin/build/loader.so
	anyopt \
		main.thorin.json network-compiled.thorin.json \
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
		--plugin plugin/build/loader.so \
		--emit-llvm \
		--log-level ${LOG_LEVEL} \
		-o combined

a.out: combined.ll allocator.cpp read.cpp
	#clang++ -O3 -L${THORIN_RUNTIME_PATH}/../build/lib -lruntime -lm $^
	clang++ -Og -g -L${THORIN_RUNTIME_PATH}/../build/lib -lruntime -lm $^
