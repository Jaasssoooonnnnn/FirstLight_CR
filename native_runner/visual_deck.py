"""Local-only card artwork and the visual deck editor's selection palette."""

from functools import lru_cache
from pathlib import Path
try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:  # Keep non-GUI tools usable without the desktop toolkit.
    tk = ttk = None

from .card_names import CARD_NAMES

ART_ROOT = Path(__file__).parent / "assets" / "cards"
KINDS = {"全部": "", "军队": "troop", "建筑": "building", "法术": "spell"}


def scrollable_tab(notebook):
    """Keep large card layouts reachable on smaller desktop windows."""
    page = ttk.Frame(notebook)
    canvas = tk.Canvas(page, highlightthickness=0, bg="#eef2f7")
    bar = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
    bar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.configure(yscrollcommand=bar.set)
    content = ttk.Frame(canvas, padding=12)
    item = canvas.create_window((0, 0), window=content, anchor="nw")
    canvas.bind("<Configure>", lambda e: canvas.itemconfigure(item, width=e.width))
    content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    def wheel(event):
        widget = event.widget
        while widget is not None:
            if widget is content:
                canvas.yview_scroll(int(-event.delta / 120), "units")
                return
            widget = getattr(widget, "master", None)
    page.winfo_toplevel().bind("<MouseWheel>", wheel, add="+")
    return page, content


@lru_cache(maxsize=1)
def card_metadata():
    from .card_specs import build_card_catalog
    return {int(s["card_id"]): s for s in build_card_catalog().to_dict()["specs"]}


def filter_cards(options, query="", kind="", cost="全部"):
    query = query.strip().casefold()
    specs = card_metadata()
    return tuple(o for o in options if
                 (not query or query in " ".join((*CARD_NAMES[o.card_id], str(o.card_id))).casefold())
                 and (not kind or specs[o.card_id]["kind"] == kind)
                 and (cost == "全部" or (float(specs[o.card_id].get("elixir_cost") or 0) >= 7
                      if cost == "7+" else float(specs[o.card_id].get("elixir_cost") or 0) == float(cost))))


def deck_summary(options, forms):
    costs = [card_metadata()[o.card_id].get("elixir_cost") for o in options if o is not None]
    known = [float(c) for c in costs if c is not None]
    avg = f"{sum(known) / len(known):.1f}" if known else "—"
    evo, hero = forms.count("觉醒"), forms.count("精英（英雄）")
    return f"平均圣水 {avg}  ·  已选 {len(costs)}/8  ·  觉醒 {evo}  ·  英雄 {hero}"


class VisualDeckMixin:
    def open_tower_palette(self):
        top = tk.Toplevel(self.frame)
        top.title("选择塔楼兵种")
        top.transient(self.frame.winfo_toplevel())
        for i, (name, cid) in enumerate(self.tower_ids.items()):
            reason = getattr(self, "disallowed_towers", {}).get(cid)
            def choose(value=name):
                self.tower_troop_var.set(value)
                top.destroy()
            ttk.Button(top, text=name.replace(" / ", "\n"), image=self._photo(cid, size=(80, 96)) or "",
                       compound="top", command=choose, state="disabled" if reason else "normal").grid(
                           row=0, column=i, padx=8, pady=(16, 4 if reason else 16))
            if reason:
                ttk.Label(top, text=reason, wraplength=150, foreground="#b91c1c").grid(
                    row=1, column=i, sticky="n", padx=8, pady=(0, 12))

    def _photo(self, card_id, form=0, size=(64, 80)):
        from PIL import Image, ImageTk
        root = self.frame.winfo_toplevel()
        if not hasattr(root, "_card_photo_cache"):
            root._card_photo_cache = {}
        cache = root._card_photo_cache
        key = (card_id, form, size)
        if key not in cache:
            path = ART_ROOT / f"{card_id}-{form}.png"
            if not form or not path.is_file():
                path = ART_ROOT / f"{card_id}.png"
            try:
                with Image.open(path) as im:
                    im = im.convert("RGBA")
                    im.thumbnail(size, Image.Resampling.LANCZOS)
                    cache[key] = ImageTk.PhotoImage(im, master=root)
            except (OSError, ValueError):
                cache[key] = None
        return cache[key]

    def _build_visual_slots(self, parent, row):
        self.active_slot = None
        self.palette = None
        self.experimental = False
        self.slot_buttons = []
        self.card_vars, self.form_vars, self.form_boxes, self.card_boxes = [], [], [], []
        self.selected_cards = [""] * 8
        grid = ttk.Frame(parent)
        grid.grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)
        for i in range(8):
            tile = ttk.Frame(grid)
            tile.grid(row=i // 4, column=i % 4, sticky="nsew", padx=4, pady=4)
            grid.columnconfigure(i % 4, weight=1, uniform="cards")
            button = tk.Button(tile, text=f"{i+1}  选择卡牌", compound="top", width=96, height=116,
                               relief="flat", bg="#ffffff", activebackground="#dbeafe",
                               highlightthickness=2, highlightbackground="#dbe3ed", wraplength=96,
                               command=lambda slot=i: self.open_palette(slot))
            button.pack(fill="x")
            self.slot_buttons.append(button)
            variable = tk.StringVar(value="")
            self.card_vars.append(variable)
            form = tk.StringVar(value="基础")
            self.form_vars.append(form)
            box = ttk.Combobox(tile, textvariable=form, values=("基础",), state="readonly", width=10)
            box.pack(fill="x", pady=(4, 0))
            self.form_boxes.append(box)
            form.trace_add("write", lambda *_: self.refresh_visuals())
        self.summary_var = tk.StringVar()
        ttk.Label(parent, textvariable=self.summary_var).grid(row=row+1, column=0, columnspan=3, sticky="w")
        self.validation_var = tk.StringVar()
        self.validation_label = ttk.Label(parent, textvariable=self.validation_var, wraplength=440)
        self.validation_label.grid(row=row+2, column=0, columnspan=3, sticky="w", pady=4)
        actions = ttk.Frame(parent)
        actions.grid(row=row+3, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Button(actions, text="浏览卡牌图鉴", command=lambda: self.open_palette()).pack(side="left")
        ttk.Button(actions, text="移除选中卡牌", command=self.remove_selected).pack(side="left", padx=4)
        self.refresh_visuals()

    def refresh_visuals(self):
        if not hasattr(self, "slot_buttons"):
            return
        options = [self.by_display.get(v.get()) for v in self.card_vars]
        forms = [v.get() for v in self.form_vars]
        duplicate_ids = {o.card_id for o in options if o and sum(p is not None and p.card_id == o.card_id for p in options) > 1}
        over = forms.count("觉醒") > 2 or forms.count("精英（英雄）") > 2 or sum(f != "基础" for f in forms) > 3
        for i, (button, option) in enumerate(zip(self.slot_buttons, options)):
            invalid = option and (option.card_id in duplicate_ids or (over and not self.experimental and forms[i] != "基础"))
            border = "#dc2626" if invalid else "#2563eb" if i == self.active_slot else "#dbe3ed"
            image = self._photo(option.card_id, {"觉醒": 1, "精英（英雄）": 2}.get(forms[i], 0)) if option else None
            text = f"{i+1}  选择卡牌" if option is None else f"{CARD_NAMES[option.card_id][0]}\n{card_metadata()[option.card_id].get('elixir_cost', '?'):g} 圣水"
            # Tk button width is in pixels only when an image is assigned.
            button.configure(image=image or "", text=text, width=96 if image else 12,
                             height=116 if image else 6, highlightbackground=border,
                             highlightcolor=border, bg="#eff6ff" if i == self.active_slot else "#ffffff")
        self.summary_var.set(deck_summary(options, forms))
        message = "同一卡组不能重复选牌。" if duplicate_ids else (
            "实验玩法：超额觉醒角色超出模型训练范围。" if over and self.experimental else
            "普通模式最多两张觉醒、两张英雄，合计三张。" if over else
            "点击卡槽替换卡牌；下方切换普通、觉醒或英雄形态。")
        tower_id = self.tower_ids.get(self.tower_troop_var.get()) if hasattr(self, "tower_troop_var") else None
        tower_reason = getattr(self, "disallowed_towers", {}).get(tower_id)
        if tower_reason:
            message = tower_reason
        self.validation_var.set(message)
        self.validation_label.configure(foreground="#b91c1c" if duplicate_ids or over or tower_reason else "#5d6b7b")
        if hasattr(self, "tower_preview"):
            cid = self.tower_ids.get(self.tower_troop_var.get())
            photo = self._photo(cid, size=(32, 40)) if cid else None
            self.tower_preview.configure(image=photo or "")

    def remove_selected(self):
        if self.active_slot is not None:
            self.card_vars[self.active_slot].set("")
            self.selected_cards[self.active_slot] = ""
            self._card_changed(self.active_slot)
            self.refresh_visuals()
            self._refresh_palette_if_open()

    def choose_card(self, option):
        selected = [self.by_display.get(v.get()) for v in self.card_vars]
        if any(o and o.card_id == option.card_id and i != self.active_slot for i, o in enumerate(selected)):
            self.palette_hint.set("这张卡已在卡组中，请选择其他卡牌。")
            return
        slot = self.active_slot
        if slot is None:
            slot = next((i for i, o in enumerate(selected) if o is None), None)
        if slot is None:
            self.palette_hint.set("卡组已满，请先在上方选择要替换的槽位。")
            return
        self.card_vars[slot].set(option.display)
        self._card_changed(slot)
        self.active_slot = next((i for i, v in enumerate(self.card_vars) if not v.get()), slot)
        self.refresh_visuals()
        self._render_palette()

    def _refresh_palette_if_open(self):
        if self.palette is not None and self.palette.winfo_exists():
            self._render_palette()

    def open_palette(self, slot=None):
        self.active_slot = slot
        self.refresh_visuals()
        if self.palette is not None and self.palette.winfo_exists():
            self.palette.lift()
            self._render_palette()
            return
        self._cancel_palette_search()
        preferences = getattr(self, "palette_preferences", None)
        if not isinstance(preferences, dict):
            preferences = self.palette_preferences = {"query": "", "kind": "全部", "cost": "全部"}
        preferences.setdefault("query", "")
        preferences.setdefault("kind", "全部")
        preferences.setdefault("cost", "全部")
        if preferences["kind"] not in KINDS:
            preferences["kind"] = "全部"
        if preferences["cost"] not in ("全部", "1", "2", "3", "4", "5", "6", "7+"):
            preferences["cost"] = "全部"
        top = self.palette = tk.Toplevel(self.frame)
        top.title("卡牌图鉴 · 点击卡牌加入或替换")
        top.geometry("860x680")
        top.minsize(720, 560)
        top.transient(self.frame.winfo_toplevel())
        self.palette_hint = tk.StringVar()
        self.palette_slots = ttk.Frame(top, padding=12)
        self.palette_slots.pack(fill="x")
        self.palette_slot_buttons = []
        for i in range(8):
            b = ttk.Button(self.palette_slots, width=18, command=lambda j=i: self._select_palette_slot(j))
            b.grid(row=i // 4, column=i % 4, sticky="ew", padx=2, pady=2)
            self.palette_slots.columnconfigure(i % 4, weight=1, uniform="palette-slots")
            self.palette_slot_buttons.append(b)
        controls = ttk.Frame(top, padding=(12, 0))
        controls.pack(fill="x")
        self.search_var = tk.StringVar(value=preferences["query"])
        self.kind_var = tk.StringVar(value=preferences["kind"])
        self.cost_var = tk.StringVar(value=preferences["cost"])
        self._palette_applied_query = self.search_var.get()
        ttk.Label(controls, text="搜索").pack(side="left")
        entry = ttk.Entry(controls, textvariable=self.search_var, width=24)
        entry.pack(side="left", fill="x", expand=True, padx=8)
        for var, choices in ((self.kind_var, tuple(KINDS)), (self.cost_var, ("全部", "1", "2", "3", "4", "5", "6", "7+"))):
            ttk.Combobox(controls, textvariable=var, values=choices, state="readonly", width=8).pack(side="left", padx=4)
            var.trace_add("write", lambda *_: self._palette_filter_changed())
        self.search_var.trace_add("write", lambda *_: self._palette_search_changed())
        ttk.Label(top, textvariable=self.palette_hint, padding=12).pack(fill="x")
        body = ttk.Frame(top)
        body.pack(fill="both", expand=True, padx=12)
        canvas = self.palette_canvas = tk.Canvas(body, bg="#eef2f7", highlightthickness=0)
        scroll = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scroll.set)
        self.palette_grid = ttk.Frame(canvas)
        self.palette_card_buttons = {}
        self.palette_empty_label = ttk.Label(
            self.palette_grid,
            text="没有匹配的卡牌，请换个关键词或筛选条件。",
            padding=24,
        )
        self.palette_empty_label.grid_remove()
        window = canvas.create_window((0, 0), window=self.palette_grid, anchor="nw")
        self.palette_grid.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        top.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))
        ttk.Button(top, text="完成配置", command=top.destroy).pack(anchor="e", padx=12, pady=8)
        top.bind("<Escape>", lambda e: top.destroy())
        self._render_palette(reset_scroll=True)
        entry.focus_set()

    def _cancel_palette_search(self):
        after_id = getattr(self, "_palette_search_after_id", None)
        if after_id is not None:
            try:
                self.palette.after_cancel(after_id)
            except (tk.TclError, AttributeError):
                pass
            self._palette_search_after_id = None

    def _palette_search_changed(self):
        self.palette_preferences["query"] = self.search_var.get()
        self._cancel_palette_search()
        self._palette_search_after_id = self.palette.after(150, self._apply_palette_search)

    def _apply_palette_search(self):
        self._palette_search_after_id = None
        self._palette_applied_query = self.search_var.get()
        self._render_palette(reset_scroll=True)

    def _palette_filter_changed(self):
        self.palette_preferences["kind"] = self.kind_var.get()
        self.palette_preferences["cost"] = self.cost_var.get()
        self._render_palette(reset_scroll=True)

    def _select_palette_slot(self, slot):
        self.active_slot = slot
        self.refresh_visuals()
        self._render_palette()

    def _render_palette(self, reset_scroll=False):
        if self.palette is None or not self.palette.winfo_exists():
            return
        scroll_position = 0 if reset_scroll else self.palette_canvas.yview()[0]
        selected = {o.card_id for v in self.card_vars if (o := self.by_display.get(v.get()))}
        for i, button in enumerate(self.palette_slot_buttons):
            option = self.by_display.get(self.card_vars[i].get())
            button.configure(text=f"{'▸' if i == self.active_slot else ''}{i+1} {CARD_NAMES[option.card_id][0] if option else '空位'}")
        cards = filter_cards(self.options, self._palette_applied_query, KINDS[self.kind_var.get()], self.cost_var.get())
        self.palette_hint.set(f"找到 {len(cards)} 张卡牌 · " + (f"正在替换第 {self.active_slot+1} 槽" if self.active_slot is not None else "点击卡牌填入空位"))
        if not cards:
            self.palette_empty_label.grid(row=0, column=0, columnspan=6, sticky="nsew")
        else:
            self.palette_empty_label.grid_remove()
        visible_ids = {option.card_id for option in cards}
        for card_id, button in self.palette_card_buttons.items():
            if card_id not in visible_ids and button.winfo_manager():
                button.grid_remove()
        for i, option in enumerate(cards):
            cost = card_metadata()[option.card_id].get("elixir_cost")
            button = self.palette_card_buttons.get(option.card_id)
            if button is None:
                button = tk.Button(self.palette_grid, compound="top", wraplength=112,
                                   relief="flat", highlightthickness=1, highlightbackground="#dbe3ed",
                                   command=lambda o=option: self.choose_card(o))
                self.palette_card_buttons[option.card_id] = button
            image = self._photo(option.card_id)
            button.configure(image=image or "",
                             text=f"{'已选 · ' if option.card_id in selected else ''}{CARD_NAMES[option.card_id][0]}\n{cost:g} 圣水",
                             bg="#dbeafe" if option.card_id in selected else "#ffffff")
            button.grid(row=i//6, column=i%6, sticky="nsew", padx=4, pady=4)
            self.palette_grid.columnconfigure(i%6, weight=1, uniform="library")
        self.palette_canvas.after_idle(lambda: self.palette_canvas.yview_moveto(scroll_position)
                                       if self.palette and self.palette.winfo_exists() else None)
