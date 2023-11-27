#include <thorin/world.h>
#include <thorin/analyses/cfg.h>
#include <thorin/analyses/scope.h>

#include <iostream>

using namespace thorin;

void * static_memory (World* world, App* app) {
    const Def* size = app->arg(1);

    Continuation* y = world->continuation(world->fn_type({
                world->mem_type(),
                world->fn_type({
                        world->mem_type(),
                        world->ptr_type(world->indefinite_array_type(world->type_qs8()))
                        })
                }));

    const Def* mem = y->param(0);
    const Def* array;

    if (auto size_lit = size->isa<PrimLit>()) {
        u64 size = size_lit->value().get_u64();
        //std::cerr << "Allocating memory of size " << size << "\n";
        auto target_type = world->ptr_type(world->indefinite_array_type(world->type_qs8()));
        const Def* inner_array = world->global(world->bottom(world->definite_array_type(world->type_qs8(), size)));
        array = world->bitcast(target_type, inner_array);
    } else {
        const Def* inner_alloca = world->alloc(world->indefinite_array_type(world->type_qs8()), mem, size);
        mem = world->extract(inner_alloca, (int) 0);
        array = world->extract(inner_alloca, 1);
    }

    y->jump(y->param(1), {mem, array});

    return y;
}

void * static_free (World* world, App* app) {
    const Def* data = app->arg(1);

    Continuation* y = world->continuation(world->fn_type({
                world->mem_type(),
                world->fn_type({
                        world->mem_type()
                        })
                }));

    const Def* mem = y->param(0);
    y->jump(y->param(1), {mem});

    return y;
}

void * build_static_array_cpp (World* world, App* app) {
    //std::cerr << "build_static_array\n";
    //app->dump();
    const Def* element = app->arg(1);
    const Def* size = app->arg(2);

    auto size_lit = size->isa<PrimLit>();
    //assert(size_lit && "Static arrays need to have a known size at this point.");
    if (!size_lit) {
        throw std::runtime_error("Static arrays need to have a known size at this point.");
    }

    u64 array_size = size_lit->value().get_u64();

    Array<const Def*> elems(array_size);

    //std::cerr << "Array Size " << array_size << "\n";

    for (u64 i = 0; i < array_size; i++) {
        elems[i] = element;
    }

    const Def* plain_array = world->definite_array(element->type(), elems);
    const Def* global_array = world->global(plain_array, false);

    const Def* result = world->bitcast(world->ptr_type(world->indefinite_array_type(element->type())), global_array);

    return const_cast<Def*>(result);
}

void * static_array_set_element_cpp (World* world, App* app) {
    //std::cerr << "static_array_set_element\n";
    //app->dump();
    const Def* element = app->arg(1);
    const Def* value = app->arg(2);

    auto lea = element->as<LEA>();

    auto index_lit = lea->index()->isa<PrimLit>();
    assert(index_lit);
    u64 index = index_lit->value().get_u64();

    auto global_bitcast = lea->ptr()->isa<Bitcast>();
    assert(global_bitcast);

    auto global = global_bitcast->from()->isa<Global>();
    assert(global);
    auto array = global->init()->isa<DefiniteArray>();
    assert(array);

    auto element_type = array->type()->elem_type();

    //array->dump();

    size_t array_size = array->num_ops();

    //std::cerr << "Array Size " << array_size << "\n";
    //std::cerr << "Index " << index << "\n";

    Array<const Def*> elems(array_size);
    for (u64 i = 0; i < array_size; i++) {
        elems[i] = array->op(i);
    }

    elems[index] = value;

    const Def* plain_array = world->definite_array(element_type, elems);
    array->replace_uses(plain_array);

    return nullptr;
}

extern "C" {

void * static_alloca (World* world, App* app) {
    return static_memory(world, app);
}

void * static_release (World* world, App* app) {
    return static_free(world, app);
}

void * build_static_array (World* world, App* app) {
    return build_static_array_cpp(world, app);
}

void * static_array_set_element (World* world, App* app) {
    return static_array_set_element_cpp(world, app);
}

}
