
void fill_fact(CombatEntityFact& f, unsigned n) {
    f.validated = n % 5 != 0;
    f.present = n % 3 != 0;
    f.visibility_validated = n % 4 != 0;
    f.native_object_id = n % 7 == 0 ? UINT32_MAX : n + 1;
    f.owner = n % 2;
    f.object_index = -int(n);
    f.secondary_index = int(n) + 1000;
    f.card_id = 26000000 + int(n);
    f.object_kind = n % 4;
    f.position_x = -int(n) * 10;
    f.position_y = int(n) * 100;
    f.invisible_count = n % 6;
}
void fill_combat(CombatRawRecord& r, unsigned n) {
    r.sequence = n + 1;
    r.generation = 7;
    r.state_epoch = UINT64_MAX;
    r.tick = n * 3;
    r.kind = static_cast<CombatRawKind>(n % 13 + 1);
    r.hook_offset = UINT64_MAX - n;
    r.caller_offset = n % 2 ? UINT64_MAX : 0;
    r.cause_sequence = n % 2 ? UINT64_MAX - 1 : 0;
    r.pool = static_cast<CombatPool>(n % 4);
    r.terminal_reason = static_cast<CombatTerminalReason>(n % 5);
    r.lethal = n % 2;
    r.deployment.validated = n % 2;
    r.deployment.deployment_sequence = UINT64_MAX;
    r.deployment.owner = n % 2;
    r.deployment.played_card_global_id = UINT32_MAX;
    r.deployment.effective_card_global_id = n % 3 ? UINT32_MAX - 1 : 0;
    r.deployment.card_parameter = UINT32_MAX - 2;
    r.deployment.deck_slot = 7;
    r.deployment.cost = 10;
    r.deployment.form_code = n % 6;
    CombatEntityFact* facts[] = {&r.target, &r.immediate_source, &r.source,
        &r.projectile, &r.related, &r.route_source_before, &r.route_target_before,
        &r.route_source_after, &r.route_target_after};
    for (unsigned i=0;i<9;++i) fill_fact(*facts[i],n+i);
    r.spawn_provenance.kind = static_cast<CombatSpawnProvenanceKind>(n % 3);
    r.spawn_provenance.hook_offset = n + 500;
    r.spawn_provenance.source_data_global_id = n % 2 ? -1 : 28000000;
    r.requested_amount = n % 2 ? kCombatUnsetInt : INT32_MAX;
    r.actual_amount = -int(n);
    r.pre_hp = n % 3 ? 1000 : kCombatUnsetInt;
    r.post_hp = 100;
    r.pre_builtin_shield = n % 3 ? 100 : kCombatUnsetInt;
    r.post_builtin_shield = 0;
    r.pre_buff_shield = n % 3 ? 30 : kCombatUnsetInt;
    r.post_buff_shield = 15;
    r.destination_before_x = n % 2 ? kCombatUnsetInt : -100;
    r.destination_before_y = n % 3 ? kCombatUnsetInt : INT32_MAX;
    r.destination_after_x = n % 3 ? kCombatUnsetInt : -200;
    r.destination_after_y = n % 2 ? kCombatUnsetInt : INT32_MAX;
}
void fill_phase(PhaseRawRecord& r, unsigned n) {
    r.sequence = n + 1;
    r.generation = 7;
    r.state_epoch = UINT64_MAX;
    r.tick = n * 3;
    r.kind = static_cast<PhaseRawKind>(n % 10 + 1);
    r.hook_offset = UINT64_MAX - n;
    r.caller_offset = n % 2 ? UINT64_MAX : 0;
    fill_fact(r.entity, n);
    fill_fact(r.target_before, n + 1);
    fill_fact(r.target_after, n + 2);
    r.buff_global_id = n % 2 ? kPhaseRuntimeUnset : -1;
    r.buff_remaining_ms = n % 2 ? kPhaseRuntimeUnset : INT32_MAX;
    r.speed_multiplier = -int(n);
    r.hit_speed_multiplier = n;
    r.input_step = n % 3 ? kPhaseRuntimeUnset : 0;
    r.output_step = n % 3 ? 100 : kPhaseRuntimeUnset;
    r.timeline_before = n * 1000;
    r.timeline_after = n * 999;
    r.classic_charge_before = n % 2 ? kPhaseRuntimeUnset : INT32_MAX;
    r.classic_charge_after = 500;
    r.success = n % 2;
}
template <class Encoder>
void emit(Encoder encoder) {
    char bytes[256000] = {};
    std::size_t used = 0;
    const bool good = encoder(bytes, sizeof(bytes), &used);
    std::cout << good << ':' << std::string(bytes, used) << '\n';
    if (good) {
        // Every bounded buffer smaller than the complete message must reject.
        for (std::size_t capacity : {std::size_t(0), std::size_t(1), used / 2, used}) {
            std::size_t short_used = 0;
            char short_bytes[256000] = {};
            assert(!encoder(short_bytes, capacity, &short_used));
            assert(short_used <= capacity);
        }
    }
}
int main() {
    g_combat_game_manager = reinterpret_cast<void*>(1);
    g_combat_epoch_active = true;
    g_combat_generation = g_phase_runtime_generation = 7;
    g_combat_state_epoch = g_phase_runtime_state_epoch = UINT64_MAX;
    g_combat_hooks_installed = g_phase_runtime_hooks_installed = true;
    g_combat_hooks_attested = g_phase_runtime_hooks_attested = true;
    for (unsigned count = 0; count < 40; ++count) {
        for (unsigned n = 0; n < count; ++n) {
            fill_combat(g_combat_event_ring[n].record, n);
            g_combat_event_ring[n].published_sequence = n + 1;
            fill_phase(g_phase_runtime_ring[n].record, n);
            g_phase_runtime_ring[n].published_sequence = n + 1;
        }
        g_combat_next_sequence = g_phase_runtime_next_sequence = count + 1;
        g_combat_hooks_installed = count % 3 != 0;
        g_phase_runtime_hooks_installed = count % 4 != 0;
        g_combat_hooks_attested = g_phase_runtime_hooks_attested = count % 5 != 0;
        g_combat_rejected_capture_count = g_phase_runtime_rejected_count = count % 7 == 0;
        emit([&](char* p, std::size_t cap, std::size_t* used) {
            return append_combat_events_json(reinterpret_cast<void*>(1), 7, UINT64_MAX, 200, p, cap, used);
        });
        emit([&](char* p, std::size_t cap, std::size_t* used) {
            return append_phase_runtime_events_json(7, UINT64_MAX, 200, p, cap, used);
        });
        NativePhaseSnapshot snapshot;
        snapshot.attack_validated = count % 2;
        snapshot.movement_validated = count % 3;
        snapshot.buffs_validated = count % 4;
        snapshot.attack_sequence_stage = count;
        snapshot.hit_speed_positive_percent = INT32_MAX;
        snapshot.hit_speed_negative_magnitude = -1;
        snapshot.attack_sequence_decay_remaining_ms = count % 3 ? 0 : kPhaseRuntimeUnset;
        emit([&](char* p, std::size_t cap, std::size_t* used) {
            return append_native_phase_snapshot_json(p, cap, used, snapshot);
        });
    }
    // Wrong identity, future event, slot overwrite and invalid entity remain fail-closed.
    for (unsigned fault=0;fault<5;++fault) {
        g_combat_hooks_installed = g_phase_runtime_hooks_installed = true;
        g_combat_next_sequence = g_phase_runtime_next_sequence = 2;
        fill_combat(g_combat_event_ring[0].record, 0);
        fill_phase(g_phase_runtime_ring[0].record, 0);
        g_combat_event_ring[0].published_sequence = g_phase_runtime_ring[0].published_sequence = 1;
        if (fault==0) g_combat_event_ring[0].record.generation = g_phase_runtime_ring[0].record.generation = 999;
        if (fault==1) g_combat_event_ring[0].record.tick = g_phase_runtime_ring[0].record.tick = 999;
        if (fault==2) g_combat_event_ring[0].published_sequence = g_phase_runtime_ring[0].published_sequence = 999;
        if (fault==3) {
            g_combat_event_ring[0].record.source.present = g_phase_runtime_ring[0].record.entity.present = true;
            g_combat_event_ring[0].record.source.native_object_id = g_phase_runtime_ring[0].record.entity.native_object_id = 0;
        }
        if (fault==4) {
            g_combat_event_ring[0].record.spawn_provenance.kind = CombatSpawnProvenanceKind::ActionSpawnToLocation;
            g_combat_event_ring[0].record.spawn_provenance.hook_offset = 0;
        }
        char bytes[256000] = {};
        std::size_t used = 0;
        bool combat = append_combat_events_json(reinterpret_cast<void*>(1), 7, UINT64_MAX, 200, bytes, sizeof(bytes), &used);
        used = 0;
        bool phase = append_phase_runtime_events_json(7, UINT64_MAX, 200, bytes, sizeof(bytes), &used);
        std::cout << "fault:" << fault << ':' << combat << ':' << phase << '\n';
    }
}
