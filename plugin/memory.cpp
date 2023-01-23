#include <thorin/world.h>
#include <thorin/analyses/cfg.h>
#include <thorin/analyses/scope.h>

#include <iostream>

using namespace thorin;

void * static_memory (void * somedef1) {
    Def* size = (Def *) somedef1;

    World& world = size->world();

    Continuation* y = world.continuation(world.fn_type({
                world.mem_type(),
                world.fn_type({
                        world.mem_type(),
                        world.ptr_type(world.indefinite_array_type(world.type_qs8()))
                        })
                }));

    const Def* mem = y->param(0);
    const Def* inner_array;

    if (auto size_lit = size->isa<PrimLit>()) {
        inner_array = world.global(world.indefinite_array(world.type_qs8(), size_lit));
    } else {
        world.dump();
        assert(false);
        const Def* inner_alloca = world.alloc(world.indefinite_array_type(world.type_qs8()), mem, size);
        mem = world.extract(inner_alloca, (int) 0);
        inner_array = world.extract(inner_alloca, 1);
    }

    y->jump(y->param(1), {mem, inner_array});

    return y;
}

void * static_free (void * somedef1) {
    Def* data = (Def *) somedef1;
    World& world = data->world();

    Continuation* y = world.continuation(world.fn_type({
                world.mem_type(),
                world.fn_type({
                        world.mem_type()
                        })
                }));

    const Def* mem = y->param(0);
    y->jump(y->param(1), {mem});

    return y;
}

extern "C" {

void * static_alloca (size_t input_c, void ** input_v) {
    //input_v[0]: size : u64
    //returns: memory buffer
    //
    //TODO: how is memory handled here?
    
    assert(input_c == 1);
    return static_memory(input_v[0]);
}

void * static_release (size_t input_c, void ** input_v) {
    //input_v[0]: buffer : &mut [i8]
    //returns: void
    //
    //TODO: how is memory handled here?
    
    assert(input_c == 1);
    return static_free(input_v[0]);
}

}
