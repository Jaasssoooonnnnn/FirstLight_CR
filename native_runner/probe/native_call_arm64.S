// AArch64 ABI bridge for libg functions that use x8 as a hidden result pointer.

    .text
    .align 2
    .global cr_call_logic_command_parse
    .type cr_call_logic_command_parse, %function

// void cr_call_logic_command_parse(
//     void* json_object,       // x0
//     void* command_context,   // x1
//     void** result,           // x2 -> x8
//     void* target);           // x3
cr_call_logic_command_parse:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    mov x8, x2
    blr x3
    ldp x29, x30, [sp], #16
    ret

    .size cr_call_logic_command_parse, .-cr_call_logic_command_parse

    .align 2
    .global cr_call_state_snapshot
    .type cr_call_state_snapshot, %function

// void cr_call_state_snapshot(
//     void* manager,          // x0
//     void** result,          // x1 -> x8
//     void* target);          // x2
cr_call_state_snapshot:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    mov x8, x1
    blr x2
    ldp x29, x30, [sp], #16
    ret

    .size cr_call_state_snapshot, .-cr_call_state_snapshot

    .align 2
    .global cr_call_card_selection_build
    .type cr_call_card_selection_build, %function

// void cr_call_card_selection_build(
//     void* battle_deck_slot, // x0
//     void* player,           // x1
//     int alternate,          // w2
//     void* result,           // x3 -> x8
//     void* target);          // x4
cr_call_card_selection_build:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    mov x8, x3
    blr x4
    ldp x29, x30, [sp], #16
    ret

    .size cr_call_card_selection_build, .-cr_call_card_selection_build

    .align 2
    .global cr_combat_heal_entry
    .type cr_combat_heal_entry, %function

// Preserve the three caller-only values needed to identify the exact heal
// producer, then forward the untouched seven-argument heal ABI to the C bridge.
//
// void cr_combat_heal_bridge(
//     x0, w1, w2, x3, w4, w5, w6,
//     caller_x19, caller_x29, caller_x30);
cr_combat_heal_entry:
    mov x9, x29
    mov x10, x30
    sub sp, sp, #32
    str x9, [sp, #0]
    str x10, [sp, #8]
    stp x29, x30, [sp, #16]
    mov x7, x19
    bl cr_combat_heal_bridge
    ldp x29, x30, [sp, #16]
    add sp, sp, #32
    ret

    .size cr_combat_heal_entry, .-cr_combat_heal_entry
