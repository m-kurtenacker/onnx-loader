.PHONY: all clean
all: a.out

.PHONY: plugin/build
plugin/build:
	@make -C plugin/build

clean:
	rm -f main.ll a.out main.thorin.json

main.thorin.json: main.art sequential.art read.art mat.art
	artic \
	      ${THORIN_RUNTIME_PATH}/artic/runtime.impala \
	      ${THORIN_RUNTIME_PATH}/artic/intrinsics_thorin.impala \
	      ${THORIN_RUNTIME_PATH}/artic/intrinsics_rv.impala \
	      ${THORIN_RUNTIME_PATH}/artic/intrinsics_math.impala \
	      ${THORIN_RUNTIME_PATH}/artic/intrinsics.impala \
	      sequential.art \
	      read.art \
	      mat.art \
	      main.art \
	      --emit-json \
	      --log-level info \
	      -o main

/*main.ll: main.art sequential.art read.art mat.art plugin/build
	artic \
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
	      -o main*/

main.ll: main.thorin.json plugin/build
	anyopt \
              main.thorin.json \
              --pass cleanup_world \
              --pass pe \
              --pass flatten_tuples \
              --pass clone_bodies \
              --pass split_slots \
              --pass plugin_execute \
              --pass lift_builtins \
              --pass inliner \
              --pass hoist_enters \
              --pass dead_load_opt \
              --pass cleanup_world \
              --pass codegen_prepare \
	      --plugin plugin/build/loader.so \
	      --emit-llvm \
	      --log-level info \
	      -o main

a.out: main.ll allocator.cpp read.cpp
	#clang++ -O3 -L${THORIN_RUNTIME_PATH}/../build/lib -lruntime -lm $^
	clang++ -Og -g -L${THORIN_RUNTIME_PATH}/../build/lib -lruntime -lm $^
