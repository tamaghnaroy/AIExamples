import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dash
from dash import dcc, html
from dash import dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

try:
    from .config import DATA_DIR, ITERATIONS_DIR, PROMPT_FILE, PROMPT_CATEGORY_FILE, TOOL_CALLS_FILE
except ImportError:
    from config import DATA_DIR, ITERATIONS_DIR, PROMPT_FILE, PROMPT_CATEGORY_FILE, TOOL_CALLS_FILE


_PROMPTS: Optional[List[Dict[str, Any]]] = None
_PROMPT_CATEGORY: Optional[Dict[str, Dict[str, Any]]] = None
_TOOLS: Optional[List[Dict[str, Any]]] = None


def _load_prompts() -> List[Dict[str, Any]]:
    global _PROMPTS
    if _PROMPTS is not None:
        return _PROMPTS

    prompts: List[Dict[str, Any]] = []
    if PROMPT_FILE.exists():
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    prompts.append(json.loads(line))
                except Exception:
                    continue

    _PROMPTS = prompts
    return prompts


def _load_prompt_category() -> Dict[str, Dict[str, Any]]:
    global _PROMPT_CATEGORY
    if _PROMPT_CATEGORY is not None:
        return _PROMPT_CATEGORY

    out: Dict[str, Dict[str, Any]] = {}
    if PROMPT_CATEGORY_FILE.exists():
        with open(PROMPT_CATEGORY_FILE, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = (row.get("prompt_id") or "").strip()
                if pid:
                    out[pid] = row

    _PROMPT_CATEGORY = out
    return out


def _load_tools() -> List[Dict[str, Any]]:
    global _TOOLS
    if _TOOLS is not None:
        return _TOOLS

    tools: List[Dict[str, Any]] = []
    if TOOL_CALLS_FILE.exists():
        with open(TOOL_CALLS_FILE, "r", encoding="utf-8") as f:
            try:
                tools = json.load(f)
            except Exception:
                tools = []

    _TOOLS = tools
    return tools


def _unique_values(prompts: List[Dict[str, Any]]) -> Tuple[List[str], List[int], List[str]]:
    categories = sorted({(p.get("category") or "").strip() for p in prompts if p.get("category")})
    difficulties = sorted({int(p.get("difficulty")) for p in prompts if str(p.get("difficulty", "")).isdigit()})
    personas = sorted({(p.get("persona") or "").strip() for p in prompts if p.get("persona")})
    return categories, difficulties, personas


def _iter_dirs() -> List[Path]:
    if not ITERATIONS_DIR.exists():
        return []

    dirs = [p for p in ITERATIONS_DIR.iterdir() if p.is_dir() and p.name.startswith("iter_")]

    def _key(x: Path) -> int:
        try:
            return int(x.name.split("_")[-1])
        except Exception:
            return 10**9

    return sorted(dirs, key=_key)


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _logs_dir() -> Path:
    return ITERATIONS_DIR / "logs"


def _list_log_files(prefix: str = "") -> List[Path]:
    d = _logs_dir()
    if not d.exists():
        return []
    files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".json"]
    if prefix:
        files = [p for p in files if p.name.startswith(prefix)]
    return sorted(files, key=lambda p: p.name)


def _filter_prompts(
    prompts: List[Dict[str, Any]],
    categories: List[str],
    difficulties: List[int],
    personas: List[str],
    q: str,
    limit: int,
) -> List[Dict[str, Any]]:
    q = (q or "").strip().lower()

    def _ok(p: Dict[str, Any]) -> bool:
        if categories and (p.get("category") not in categories):
            return False
        if difficulties and (int(p.get("difficulty", 0)) not in difficulties):
            return False
        if personas and (p.get("persona") not in personas):
            return False
        if q:
            txt = (p.get("prompt_text") or "").lower()
            if q not in txt and q not in (p.get("prompt_id") or "").lower():
                return False
        return True

    out = [p for p in prompts if _ok(p)]
    if limit > 0:
        out = out[:limit]
    return out


def create_app() -> dash.Dash:
    prompts = _load_prompts()
    categories, difficulties, personas = _unique_values(prompts)

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )

    app.layout = dbc.Container(
        fluid=True,
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        width=3,
                        children=[
                            html.H4("Prompt Universe Explorer", className="mt-3"),
                            html.Hr(),
                            dbc.Label("Category"),
                            dcc.Dropdown(
                                id="pu-category",
                                options=[{"label": c, "value": c} for c in categories],
                                multi=True,
                                placeholder="All categories",
                            ),
                            dbc.Label("Difficulty", className="mt-2"),
                            dcc.Dropdown(
                                id="pu-difficulty",
                                options=[{"label": str(d), "value": d} for d in difficulties],
                                multi=True,
                                placeholder="All difficulties",
                            ),
                            dbc.Label("Persona", className="mt-2"),
                            dcc.Dropdown(
                                id="pu-persona",
                                options=[{"label": p, "value": p} for p in personas],
                                multi=True,
                                placeholder="All personas",
                            ),
                            dbc.Label("Search", className="mt-2"),
                            dbc.Input(id="pu-search", type="text", placeholder="Search prompt text / id"),
                            dbc.Label("Max rows", className="mt-2"),
                            dbc.Input(
                                id="pu-limit",
                                type="number",
                                min=100,
                                step=100,
                                value=2000,
                            ),
                            html.Div(id="pu-count", className="mt-2 text-muted"),
                        ],
                    ),
                    dbc.Col(
                        width=9,
                        children=[
                            dbc.Tabs(
                                id="pu-tabs",
                                active_tab="tab-prompts",
                                children=[
                                    dbc.Tab(label="Prompts", tab_id="tab-prompts"),
                                    dbc.Tab(label="Tools", tab_id="tab-tools"),
                                    dbc.Tab(label="Iterations", tab_id="tab-iterations"),
                                    dbc.Tab(label="LLM Logs", tab_id="tab-logs"),
                                ],
                                className="mt-3",
                            ),
                            html.Div(id="pu-tab-content", className="mt-3"),
                        ],
                    ),
                ]
            )
        ],
    )

    @app.callback(Output("pu-tab-content", "children"), Input("pu-tabs", "active_tab"))
    def _render_tab(active_tab: str):
        if active_tab == "tab-tools":
            return html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                width=4,
                                children=[
                                    dbc.Label("Tool name contains"),
                                    dbc.Input(id="pu-tool-search", type="text", placeholder="e.g. pricing"),
                                    dbc.Label("Max rows", className="mt-2"),
                                    dbc.Input(id="pu-tool-limit", type="number", min=50, step=50, value=500),
                                ],
                            ),
                            dbc.Col(width=8, children=[html.Div(id="pu-tool-count", className="text-muted")]),
                        ]
                    ),
                    dash_table.DataTable(
                        id="pu-tools-table",
                        columns=[
                            {"name": "tool_name", "id": "tool_name"},
                            {"name": "tool_description", "id": "tool_description"},
                        ],
                        data=[],
                        page_size=15,
                        sort_action="native",
                        row_selectable="single",
                        style_cell={"textAlign": "left", "whiteSpace": "normal", "height": "auto"},
                        style_table={"overflowX": "auto"},
                    ),
                    html.Hr(),
                    html.H5("Tool details"),
                    html.Pre(id="pu-tool-details", style={"maxHeight": "420px", "overflowY": "auto"}),
                ]
            )

        if active_tab == "tab-iterations":
            iters = _iter_dirs()
            opts = [{"label": d.name, "value": d.name} for d in iters]
            value = opts[-1]["value"] if opts else None
            return html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                width=4,
                                children=[
                                    dbc.Label("Iteration"),
                                    dcc.Dropdown(id="pu-iter", options=opts, value=value, clearable=False),
                                ],
                            ),
                            dbc.Col(width=8, children=[html.Div(id="pu-iter-summary", className="text-muted")]),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(width=6, children=[html.H5("stats.json"), html.Pre(id="pu-iter-stats")]),
                            dbc.Col(width=6, children=[html.H5("prompt_evaluation.json"), html.Pre(id="pu-iter-prompt-eval")]),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(width=12, children=[html.H5("tool_evaluation.json"), html.Pre(id="pu-iter-tool-eval", style={"maxHeight": "480px", "overflowY": "auto"})]),
                        ]
                    ),
                ]
            )

        if active_tab == "tab-logs":
            files = _list_log_files(prefix="")
            opts = [{"label": p.name, "value": p.name} for p in files]
            value = opts[-1]["value"] if opts else None
            return html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                width=3,
                                children=[
                                    dbc.Label("Log type"),
                                    dcc.Dropdown(
                                        id="pu-log-prefix",
                                        options=[
                                            {"label": "All", "value": ""},
                                            {"label": "prompt_eval", "value": "prompt_eval"},
                                            {"label": "tool_eval", "value": "tool_eval"},
                                            {"label": "tool_validation", "value": "tool_validation"},
                                        ],
                                        value="",
                                        clearable=False,
                                    ),
                                ],
                            ),
                            dbc.Col(
                                width=9,
                                children=[
                                    dbc.Label("Log file"),
                                    dcc.Dropdown(id="pu-log-file", options=opts, value=value, clearable=False),
                                ],
                            ),
                        ]
                    ),
                    html.Hr(),
                    html.H5("Request (system/user prompt)") ,
                    html.Pre(id="pu-log-request", style={"maxHeight": "320px", "overflowY": "auto"}),
                    html.H5("Response") ,
                    html.Pre(id="pu-log-response", style={"maxHeight": "420px", "overflowY": "auto"}),
                ]
            )

        return html.Div(
            [
                dash_table.DataTable(
                    id="pu-prompts-table",
                    columns=[
                        {"name": "prompt_id", "id": "prompt_id"},
                        {"name": "category", "id": "category"},
                        {"name": "difficulty", "id": "difficulty"},
                        {"name": "persona", "id": "persona"},
                        {"name": "prompt_text", "id": "prompt_text"},
                    ],
                    data=[],
                    page_size=15,
                    sort_action="native",
                    filter_action="none",
                    row_selectable="single",
                    style_cell={"textAlign": "left", "whiteSpace": "normal", "height": "auto", "maxWidth": 420},
                    style_table={"overflowX": "auto"},
                ),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(width=6, children=[html.H5("Prompt"), html.Pre(id="pu-prompt-details", style={"maxHeight": "420px", "overflowY": "auto"})]),
                        dbc.Col(width=6, children=[html.H5("Matching tools"), html.Pre(id="pu-matching-tools", style={"maxHeight": "420px", "overflowY": "auto"})]),
                    ]
                ),
            ]
        )

    @app.callback(
        Output("pu-prompts-table", "data"),
        Output("pu-count", "children"),
        Input("pu-category", "value"),
        Input("pu-difficulty", "value"),
        Input("pu-persona", "value"),
        Input("pu-search", "value"),
        Input("pu-limit", "value"),
    )
    def _update_prompt_table(cat_v, diff_v, persona_v, q, limit):
        ps = _load_prompts()
        cats = cat_v or []
        diffs = diff_v or []
        pers = persona_v or []
        try:
            lim = int(limit) if limit is not None else 2000
        except Exception:
            lim = 2000
        filtered = _filter_prompts(ps, cats, diffs, pers, q or "", lim)

        rows: List[Dict[str, Any]] = []
        for p in filtered:
            rows.append(
                {
                    "prompt_id": p.get("prompt_id"),
                    "category": p.get("category"),
                    "difficulty": p.get("difficulty"),
                    "persona": p.get("persona"),
                    "prompt_text": p.get("prompt_text"),
                }
            )

        return rows, f"Showing {len(rows)} prompts (limit={lim})"

    @app.callback(
        Output("pu-prompt-details", "children"),
        Output("pu-matching-tools", "children"),
        Input("pu-prompts-table", "selected_rows"),
        State("pu-prompts-table", "data"),
    )
    def _update_prompt_details(selected_rows, table_data):
        if not selected_rows or not table_data:
            return "", ""

        idx = selected_rows[0]
        if idx >= len(table_data):
            return "", ""

        pid = table_data[idx].get("prompt_id")
        all_prompts = _load_prompts()
        p = next((x for x in all_prompts if x.get("prompt_id") == pid), None)
        if not p:
            return "", ""

        cat = p.get("category") or ""
        tools = _load_tools()
        matching = [t for t in tools if (t.get("tool_name") == cat) or (cat and cat in (t.get("tool_name") or ""))]
        if not matching and cat:
            matching = [t for t in tools if cat.lower() in (t.get("tool_description") or "").lower()]

        cat_map = _load_prompt_category()
        extra = cat_map.get(pid, {})

        prompt_view = {
            **p,
            "difficulty_rationale": extra.get("difficulty_rationale", ""),
            "category_rationale": extra.get("category_rationale", ""),
        }

        tools_view = matching[:50]

        return json.dumps(prompt_view, indent=2, ensure_ascii=False), json.dumps(tools_view, indent=2, ensure_ascii=False)

    @app.callback(
        Output("pu-tools-table", "data"),
        Output("pu-tool-count", "children"),
        Input("pu-tool-search", "value"),
        Input("pu-tool-limit", "value"),
    )
    def _update_tools_table(q, limit):
        tools = _load_tools()
        q = (q or "").strip().lower()
        try:
            lim = int(limit) if limit is not None else 500
        except Exception:
            lim = 500

        out: List[Dict[str, Any]] = []
        for t in tools:
            name = (t.get("tool_name") or "")
            desc = (t.get("tool_description") or "")
            if q and q not in name.lower() and q not in desc.lower():
                continue
            out.append({"tool_name": name, "tool_description": desc})
            if lim > 0 and len(out) >= lim:
                break

        return out, f"Showing {len(out)} tools (limit={lim})"

    @app.callback(
        Output("pu-tool-details", "children"),
        Input("pu-tools-table", "selected_rows"),
        State("pu-tools-table", "data"),
    )
    def _update_tool_details(selected_rows, table_data):
        if not selected_rows or not table_data:
            return ""
        idx = selected_rows[0]
        if idx >= len(table_data):
            return ""
        name = table_data[idx].get("tool_name")
        tools = _load_tools()
        tool = next((t for t in tools if t.get("tool_name") == name), None)
        if not tool:
            return ""
        return json.dumps(tool, indent=2, ensure_ascii=False)

    @app.callback(
        Output("pu-iter-summary", "children"),
        Output("pu-iter-stats", "children"),
        Output("pu-iter-prompt-eval", "children"),
        Output("pu-iter-tool-eval", "children"),
        Input("pu-iter", "value"),
    )
    def _update_iteration(iter_name: Optional[str]):
        if not iter_name:
            return "", "", "", ""

        d = ITERATIONS_DIR / iter_name
        stats = _safe_read_json(d / "stats.json")
        pe = _safe_read_json(d / "prompt_evaluation.json")
        te = _safe_read_json(d / "tool_evaluation.json")

        summary = ""
        if stats:
            summary = (
                f"prompts={stats.get('total_prompts')}, tools={stats.get('total_tools')}, "
                f"score={stats.get('overall_score')}, tool_score={stats.get('tool_score')}"
            )

        return (
            summary,
            json.dumps(stats or {}, indent=2, ensure_ascii=False),
            json.dumps(pe or {}, indent=2, ensure_ascii=False),
            json.dumps(te or {}, indent=2, ensure_ascii=False),
        )

    @app.callback(
        Output("pu-log-file", "options"),
        Output("pu-log-file", "value"),
        Input("pu-log-prefix", "value"),
    )
    def _update_log_file_list(prefix: str):
        files = _list_log_files(prefix=prefix or "")
        opts = [{"label": p.name, "value": p.name} for p in files]
        value = opts[-1]["value"] if opts else None
        return opts, value

    @app.callback(
        Output("pu-log-request", "children"),
        Output("pu-log-response", "children"),
        Input("pu-log-file", "value"),
    )
    def _show_log_file(file_name: Optional[str]):
        if not file_name:
            return "", ""

        p = _logs_dir() / file_name
        obj = _safe_read_json(p)
        if not obj:
            return "", ""

        req = obj.get("request", {})
        resp = obj.get("response", {})
        return json.dumps(req, indent=2, ensure_ascii=False), json.dumps(resp, indent=2, ensure_ascii=False)

    return app


def run_ui(host: str = "127.0.0.1", port: int = 8051, debug: bool = True):
    app = create_app()
    run_fn = getattr(app, "run", None)
    if callable(run_fn):
        run_fn(debug=debug, host=host, port=port)
        return
    app.run_server(debug=debug, host=host, port=port)


if __name__ == "__main__":
    run_ui()
