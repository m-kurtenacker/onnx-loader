.PHONY: all clean
all: a.out

.PHONY: plugin/build
plugin/build:
	@make -C plugin/build

clean:
	rm -f main.ll a.out

main.ll: main.art sequential.art read.art mat.art plugin/build
	artic --emit-llvm \
	      /home/matthiask/Documents/anydsl-networks/runtime/platforms/artic/runtime.impala \
	      /home/matthiask/Documents/anydsl-networks/runtime/platforms/artic/intrinsics_thorin.impala \
	      /home/matthiask/Documents/anydsl-networks/runtime/platforms/artic/intrinsics_rv.impala \
	      /home/matthiask/Documents/anydsl-networks/runtime/platforms/artic/intrinsics_math.impala \
	      /home/matthiask/Documents/anydsl-networks/runtime/platforms/artic/intrinsics.impala \
	      sequential.art \
	      read.art \
	      mat.art \
	      main.art \
	      --plugin plugin/build/loader.so \
	      --emit-llvm \
	      --log-level info \
	      -o main

a.out: main.ll allocator.cpp read.cpp
	#clang++ -O3 -L/home/matthiask/Documents/anydsl-networks/runtime/build/lib -lruntime -lm $^
	clang++ -Og -g -L/home/matthiask/Documents/anydsl-networks/runtime/build/lib -lruntime -lm $^
