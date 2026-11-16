import pytest

tk = pytest.importorskip("tkinter")

from native_runner.user_interface import DeckEditor, load_card_options, BUILTIN_DECKS
from native_runner.visual_deck import filter_cards, deck_summary


@pytest.fixture(scope="module")
def tk_root(desktop_root):
    return desktop_root


@pytest.fixture
def editor(tk_root):
    editor = DeckEditor(tk_root, title="测试卡组", options=load_card_options(),
                        on_save=lambda e: None, on_load=lambda e: None, on_delete=lambda e: None)
    yield editor
    if editor.palette is not None and editor.palette.winfo_exists():
        editor.palette.destroy()
    editor.frame.destroy()


def test_filters_support_chinese_english_type_and_cost():
    options = load_card_options()
    assert filter_cards(options, "野猪骑士")[0].card_id == 26000021
    assert filter_cards(options, "HOG RIDER")[0].card_id == 26000021
    assert filter_cards(options, "26000021")[0].card_id == 26000021
    assert not filter_cards(options, "野猪骑士", kind="spell")
    assert not filter_cards(options, "野猪骑士", cost="2")
    assert filter_cards(options, "野猪骑士", cost="4")
    assert not filter_cards(options, "not a real card")


def test_visual_preset_round_trip_and_role_feedback(editor):
    preset = BUILTIN_DECKS[0]
    editor.set_deck(preset.deck, preset.forms, preset.tower_troop_id)
    assert editor.values() == (preset.deck, preset.forms)
    assert editor.tower_troop_id() == preset.tower_troop_id
    assert "8/8" in editor.summary_var.get()
    assert "最多" not in editor.validation_var.get()
    assert editor._photo(preset.deck[0]) is not None
    assert editor._photo(-1) is None


def test_palette_replaces_only_selected_slot_and_blocks_duplicates(editor):
    preset = BUILTIN_DECKS[0]
    editor.set_deck(preset.deck, preset.forms)
    editor.open_palette(0)
    editor.palette.withdraw()
    duplicate = editor.by_id[preset.deck[1]]
    editor.choose_card(duplicate)
    assert editor.values()[0] == preset.deck
    assert "已在卡组" in editor.palette_hint.get()
    replacement = next(o for o in editor.options if o.card_id not in preset.deck)
    editor.choose_card(replacement)
    assert editor.values()[0] == (replacement.card_id, *preset.deck[1:])
    editor.remove_selected()
    assert not editor.card_vars[0].get()
    editor.active_slot = None
    editor.choose_card(replacement)
    assert editor.values()[0][0] == replacement.card_id
    assert len(editor.palette_card_buttons) == len(editor.options)


def test_empty_summary_has_no_fake_average():
    summary = deck_summary([None] * 8, ["基础"] * 8)
    assert "0/8" in summary and "—" in summary


def test_role_limit_marks_slots_and_experimental_mode_warns(editor):
    options = [o for o in editor.options if o.supports_evolution][:8]
    editor.set_deck(tuple(o.card_id for o in options), (1,) * 8)
    assert "最多" in editor.validation_var.get()
    assert editor.slot_buttons[0].cget("highlightbackground") == "#dc2626"
    editor.experimental = True
    editor.refresh_visuals()
    assert "实验玩法" in editor.validation_var.get()


def test_palette_debounces_search_reuses_cards_and_keeps_empty_state(editor, monkeypatch, tk_root):
    editor.palette_preferences = {"query": "", "kind": "全部", "cost": "全部"}
    monkeypatch.setattr(editor, "_photo", lambda *args, **kwargs: None)
    editor.open_palette()
    editor.palette.withdraw()
    tk_root.update_idletasks()
    card_id = editor.options[0].card_id
    button = editor.palette_card_buttons[card_id]
    scheduled = []
    monkeypatch.setattr(editor.palette, "after", lambda delay, callback: scheduled.append((delay, callback)) or "search")
    assert button.cget("image") == ""
    assert button.cget("text")

    editor.search_var.set("not a real card")
    assert scheduled[-1][0] == 150
    assert "找到 0 张" not in editor.palette_hint.get()
    assert editor.palette_preferences["query"] == "not a real card"
    scheduled[-1][1]()

    assert editor.palette_hint.get().startswith("找到 0 张")
    assert editor.palette_empty_label.winfo_manager() == "grid"
    assert editor.palette_card_buttons[card_id] is button
    assert button.cget("text")


def test_palette_preserves_scroll_for_selection_and_slot_changes(editor, tk_root):
    editor.open_palette()
    editor.palette.withdraw()
    tk_root.update_idletasks()
    canvas = editor.palette_canvas
    canvas.yview_moveto(0.55)
    tk_root.update_idletasks()
    before = canvas.yview()[0]
    assert before > 0.1
    selected_card_button = editor.palette_card_buttons[editor.options[0].card_id]

    editor.choose_card(editor.options[0])
    tk_root.update_idletasks()
    after_choose = canvas.yview()[0]
    assert abs(after_choose - before) < 0.05
    assert editor.palette_card_buttons[editor.options[0].card_id] is selected_card_button

    editor._select_palette_slot(4)
    tk_root.update_idletasks()
    assert abs(canvas.yview()[0] - after_choose) < 0.05

    editor.kind_var.set("建筑")
    tk_root.update_idletasks()
    assert canvas.yview()[0] == 0


def test_disallowed_tower_stays_selected_with_reason(editor):
    tower_name, tower_id = next(iter(editor.tower_ids.items()))
    reason = "该塔兵尚未验证，请选择其他塔兵。"
    editor.disallowed_towers = {tower_id: reason}
    editor.tower_troop_var.set(tower_name)
    editor.refresh_visuals()
    assert editor.tower_troop_var.get() == tower_name
    assert editor.validation_var.get() == reason


def test_disallowed_tower_palette_button_is_disabled_with_reason(editor, tk_root):
    _, tower_id = next(iter(editor.tower_ids.items()))
    reason = "该塔兵尚未验证，请选择其他塔兵。"
    editor.disallowed_towers = {tower_id: reason}
    editor.open_tower_palette()
    palette = next(w for w in editor.frame.winfo_children()
                   if isinstance(w, tk.Toplevel) and w.title() == "选择塔楼兵种")
    button = palette.grid_slaves(row=0, column=0)[0]
    label = palette.grid_slaves(row=1, column=0)[0]
    assert str(button.cget("state")) == "disabled"
    assert label.cget("text") == reason
    palette.destroy()


def test_complete_interface_builds_without_starting_workers(tk_root, monkeypatch, tmp_path):
    from native_runner.user_interface import CRHarnessInterface
    monkeypatch.setattr(CRHarnessInterface, "_background", lambda *a, **k: None)
    monkeypatch.setattr(CRHarnessInterface, "refresh_collected_replays", lambda self: None)
    app = CRHarnessInterface(tk_root, preflight=None, log_path=tmp_path / "preview.log")
    assert len(app.deck_editors) == 6
    assert app.duel_mode_box.get()
    assert callable(app.apply_duel_preset)

