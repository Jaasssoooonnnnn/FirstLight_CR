import pytest

tk = pytest.importorskip("tkinter")

from native_runner.user_interface import DeckEditor, load_card_options, BUILTIN_DECKS
from native_runner.visual_deck import filter_cards, deck_summary


@pytest.fixture(scope="module")
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as error:
        if "display" in str(error).lower():
            pytest.skip("A desktop display is required for Tk widget tests")
        raise
    root.withdraw()
    yield root
    root.destroy()


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
    assert len(editor.palette_grid.winfo_children()) == len(editor.options)


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


def test_complete_interface_builds_without_starting_workers(tk_root, monkeypatch, tmp_path):
    from native_runner.user_interface import CRHarnessInterface
    monkeypatch.setattr(CRHarnessInterface, "_background", lambda *a, **k: None)
    monkeypatch.setattr(CRHarnessInterface, "refresh_collected_replays", lambda self: None)
    app = CRHarnessInterface(tk_root, preflight=None, log_path=tmp_path / "preview.log")
    assert len(app.deck_editors) == 6
    assert app.duel_mode_box.get()
    assert callable(app.apply_duel_preset)
