
NativeEntityReferenceView ref(int seed) {
    NativeEntityReferenceView r{};
    r.validated = seed % 3 != 0; r.has_value = seed % 5 != 0;
    r.native_object_id = 4000000000U + seed;
    r.owner = seed % 2; r.object_index = seed + 1; r.secondary_index = seed + 4;
    return r;
}
template <typename Fn, typename... Args> void emit(Fn fn, Args... args) {
    char buffer[65536] = {}; std::size_t used = 0;
    bool ok = fn(buffer, sizeof(buffer), &used, args...);
    std::printf("%d\t%s\n",ok,buffer);
    for (std::size_t capacity: {std::size_t(1), std::size_t(17), used, used+1}) {
        char small[65536] = {}; std::size_t count = 0;
        bool fit = fn(small,capacity,&count,args...);
        std::printf("capacity:%zu:%d\n",capacity,fit);
    }
}
int main() {
    for (int seed=0;seed<180;++seed) {
        NativeCaptureRuntimeStateView c{};
        c.action_data_global_id=UINT32_MAX-seed; c.drag_delay_ms=seed;
        c.grab_pause_ms=seed*2;c.capture_drag_time_ms=seed*3;
        c.configured_cooldown_ms=seed*4;c.cooldown_remaining_ms=seed%4;
        c.hit_frequency_ms=seed*5;c.hit_accumulator_ms=seed*6;
        c.completion_result_current_update=seed%2;c.first_capture_handled=seed%3;
        c.target_count=seed%4;
        for (int i=0;i<c.target_count;++i) {
            auto& t=c.targets[i];t.native_object_id=UINT32_MAX-i;t.entity=ref(seed+i);
            t.elapsed_ms=-seed;t.phase_budget_remaining_ms=seed*7;
            t.phase_budget_remaining_known=(seed+i)%2;t.phase=static_cast<NativeCaptureTargetPhase>((seed+i)%5);
        }
        emit(append_native_capture_runtime_json,c);
        NativeThresholdRelocationStateView r{};
        r.action_data_global_id=UINT32_MAX-seed;r.phase=static_cast<NativeThresholdRelocationPhase>(seed%3);
        r.stage=seed%7;r.relocation_index=seed%3;r.hide_duration_ms=seed*23;
        r.remaining_ms=seed*17;r.burrowed=seed%2;r.threshold_count=seed%4;
        for(int i=0;i<r.threshold_count;++i)r.thresholds[i]=90-i*30;
        emit(append_native_threshold_relocation_json,r);
        NativeProjectileStateView p{};
        p.data_global_id=UINT32_MAX-seed;p.source=ref(seed);p.target=ref(seed+1);p.homing_target=ref(seed+2);
        p.destination_x=INT32_MIN+seed;p.destination_y=INT32_MAX-seed;p.terminal=seed%2;p.drag_stage=seed%3-1;
        emit(append_native_projectile_json,p);
        NativeEntityResourceStateView e{};
        e.current_raw=seed;e.capacity_raw=1+seed*7;e.base_raw=seed*2;e.limit_raw=seed*14;
        emit(append_native_entity_resource_json,e);
        NativePeriodicAttackModifierStateView m{};
        m.action_data_global_id=UINT32_MAX-seed;m.phase=static_cast<NativePeriodicAttackModifierPhase>(seed%2);
        m.period_attacks=seed%4;m.completed_attacks=seed%3;m.added_damage_raw=seed*17;
        m.linger_duration_ms=seed*23;m.linger_remaining_ms=seed*11;m.source_native_object_id=UINT32_MAX-seed;m.source=ref(seed);
        emit(append_native_periodic_attack_modifier_json,m);
        NativePlayerRuntimeView player{};
        player.owner_root_valid=seed%3;player.owner_root=ref(seed);
        player.ability_runtime_valid=seed%4;player.ability_count=seed%3;
        for(int i=0;i<player.ability_count;++i){auto& a=player.abilities[i];
            a.controller_slot=i;a.action_data_global_id=1234+seed;
            std::strcpy(a.action_data_name,"ability \\\"\n\t\xe4\xb8\xad");a.action_data_name_size=std::strlen(a.action_data_name);
            a.selected_character_global_id=123456+seed;a.remaining_cooldown_ms=seed;
            a.configured_cooldown_ms=seed*7;a.remaining_charges_raw=seed%3-1;a.max_charges=seed%3;
            a.button_state=(seed+i)%15;a.champion_count=seed%3;
            for(int j=0;j<a.champion_count;++j)a.champions[j]=ref(seed+j);
        }
        player.evolution_runtime_valid=seed%5;player.evolution_slot_count=seed%9;
        for(int i=0;i<player.evolution_slot_count;++i){auto& e=player.evolution_slots[i];
            e.deck_slot=i;e.card_id=26000000+i;e.base_spell_global_id=seed*100+i;
            e.evolvable=(seed+i)%2;e.evolution_form_global_id=seed*100+i+2;
            e.progress=seed%3;e.cycle_required=2;e.cycle_remaining=seed%3;e.ready=seed%2;
        }
        emit(append_native_player_runtime_json,seed%2,player);
    }
}
