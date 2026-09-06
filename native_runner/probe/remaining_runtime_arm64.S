.text
.p2align 2

// The resource producers need the original caller LR and its x19 source.
// Tail-branching into the C bridge preserves the native caller's LR while the
// bridge calls the relocated original with the unmodified x0/w1/w2 ABI.
.global remaining_resource_whole_entry
.type remaining_resource_whole_entry, %function
remaining_resource_whole_entry:
    mov x3, x30
    mov x4, x19
    b remaining_resource_whole_bridge
.size remaining_resource_whole_entry, .-remaining_resource_whole_entry

.global remaining_resource_fixed_entry
.type remaining_resource_fixed_entry, %function
remaining_resource_fixed_entry:
    mov x3, x30
    mov x4, x19
    b remaining_resource_fixed_bridge
.size remaining_resource_fixed_entry, .-remaining_resource_fixed_entry

// The SpawnAttach helper has additional/unknown ABI arguments. Preserve all
// integer and SIMD argument registers plus x8, retain the exact call/data
// context, then restore the exact original ABI and tail-branch to the helper.
.global remaining_spawn_helper_entry
.type remaining_spawn_helper_entry, %function
remaining_spawn_helper_entry:
    sub sp, sp, #0xd0
    stp x0, x1, [sp, #0x00]
    stp x2, x3, [sp, #0x10]
    stp x4, x5, [sp, #0x20]
    stp x6, x7, [sp, #0x30]
    stp x8, x30, [sp, #0x40]
    stp q0, q1, [sp, #0x50]
    stp q2, q3, [sp, #0x70]
    stp q4, q5, [sp, #0x90]
    stp q6, q7, [sp, #0xb0]
    mov x3, x30
    bl remaining_spawn_helper_begin_bridge
    ldp q6, q7, [sp, #0xb0]
    ldp q4, q5, [sp, #0x90]
    ldp q2, q3, [sp, #0x70]
    ldp q0, q1, [sp, #0x50]
    ldp x8, x30, [sp, #0x40]
    ldp x6, x7, [sp, #0x30]
    ldp x4, x5, [sp, #0x20]
    ldp x2, x3, [sp, #0x10]
    ldp x0, x1, [sp, #0x00]
    add sp, sp, #0xd0
    adrp x16, :got:g_original_remaining_spawn_helper
    ldr x16, [x16, :got_lo12:g_original_remaining_spawn_helper]
    ldr x16, [x16]
    br x16
.size remaining_spawn_helper_entry, .-remaining_spawn_helper_entry

// f2ffb8 is a normal function entry reached through the object-manager
// vtable. Preserve the complete native argument ABI, while passing the exact
// caller LR, ambient source (x21), and f24188's stable request+0x48 stack
// local to the C observer. The observer only records child-pointer provenance;
// the relocated original remains solely responsible for native execution.
.global remaining_action_spawn_mode1_child_create_entry
.type remaining_action_spawn_mode1_child_create_entry, %function
remaining_action_spawn_mode1_child_create_entry:
    sub sp, sp, #0xd0
    stp x0, x1, [sp, #0x00]
    stp x2, x3, [sp, #0x10]
    stp x4, x5, [sp, #0x20]
    stp x6, x7, [sp, #0x30]
    stp x8, x30, [sp, #0x40]
    stp q0, q1, [sp, #0x50]
    stp q2, q3, [sp, #0x70]
    stp q4, q5, [sp, #0x90]
    stp q6, q7, [sp, #0xb0]
    mov x4, x30
    mov x5, x21
    ldr x6, [sp, #0x138]
    bl remaining_action_spawn_mode1_child_create_bridge
    ldp q6, q7, [sp, #0xb0]
    ldp q4, q5, [sp, #0x90]
    ldp q2, q3, [sp, #0x70]
    ldp q0, q1, [sp, #0x50]
    ldp x8, x30, [sp, #0x40]
    ldp x6, x7, [sp, #0x30]
    ldp x4, x5, [sp, #0x20]
    ldp x2, x3, [sp, #0x10]
    ldp x0, x1, [sp, #0x00]
    add sp, sp, #0xd0
    adrp x16, :got:g_original_remaining_action_spawn_mode1_child_create
    ldr x16, [x16, :got_lo12:g_original_remaining_action_spawn_mode1_child_create]
    ldr x16, [x16]
    br x16
.size remaining_action_spawn_mode1_child_create_entry, .-remaining_action_spawn_mode1_child_create_entry

.section .note.GNU-stack,"",%progbits
