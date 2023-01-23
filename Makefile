.PHONY: all clean
all: a.out

.PHONY: plugin/build
plugin/build:
	@make -C plugin/build

clean:
	rm -f main.ll a.out

main.ll: main.art sequential.art read.art mat.art plugin/build
	artic --emit-llvm \
	      ${THORIN_RUNTIME_PATH}/artic/runtime.impala \
	      ${THORIN_RUNTIME_PATH}/artic/intrinsics_thorin.impala \
	      ${THORIN_RUNTIME_PATH}/artic/intrinsics_rv.impala \
	      ${THORIN_RUNTIME_PATH}/artic/intrinsics_math.impala \
	      ${THORIN_RUNTIME_PATH}/artic/intrinsics.impala \
	      sequential.art \
	      read.art \
	      mat.art \
	      main.art \
	      --plugin plugin/build/loader.so \
	      --emit-llvm \
	      --log-level info \
	      -o main

a.out: main.ll allocator.cpp read.cpp
	#clang++ -O3 -L${THORIN_RUNTIME_PATH}/../build/lib -lruntime -lm $^
	clang++ -Og -g -L${THORIN_RUNTIME_PATH}/../build/lib -lruntime -lm $^
