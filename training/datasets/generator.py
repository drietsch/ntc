"""Deterministic mini synthetic dataset generator (V1 mini-scale stand-in for
the full teacher-driven engine of spec §51–§53).

Utterances are composed from per-language phrase banks with gold spans tracked
by construction, so every example is valid-by-construction against
`datasets.schema.DatasetExample`. Covers: CALL/ASK/NO_CALL, span/enum/
boolean/integer/duration/datetime values, decoy candidates, mention-vs-request
hard negatives, tool/argument name randomization (spec §49), and
unseen-family + masked-name eval splits (spec §62).

The live-teacher path (`synthetic.orchestrator` driving `claude -p`) remains
the production engine; this generator provides the deterministic base corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from datasets.schema import DatasetExample

Part = str | tuple[str, str]  # literal, or (text, slot_key)

# --------------------------------------------------------------------------
# Tool families (raw schemas, flat NTC style)
# --------------------------------------------------------------------------

TRAIN_TOOLS: dict[str, dict[str, Any]] = {
    "calendar.create": {
        "name": "calendar.create",
        "description": "Create a calendar event",
        "parameters": {
            "title": {"type": "string", "description": "event title", "required": True},
            "start": {"type": "string", "format": "date-time", "required": True},
            "duration_minutes": {
                "type": "integer",
                "semantic": "DURATION",
                "description": "length of the event",
            },
        },
    },
    "email.send": {
        "name": "email.send",
        "description": "Send an email to somebody",
        "parameters": {
            "recipient": {"type": "string", "description": "who receives it", "required": True},
            "subject": {"type": "string", "description": "what it is about"},
        },
    },
    "timer.set": {
        "name": "timer.set",
        "description": "Start a countdown timer",
        "parameters": {
            "length": {
                "type": "string",
                "format": "duration",
                "semantic": "DURATION",
                "description": "how long the timer runs",
                "required": True,
            },
            "label": {"type": "string", "description": "name of the timer"},
        },
    },
    "weather.lookup": {
        "name": "weather.lookup",
        "description": "Look up the weather forecast for a place",
        "parameters": {
            "city": {"type": "location", "description": "place to check", "required": True},
            "days": {"type": "integer", "description": "how many days ahead"},
        },
    },
    "light.set": {
        "name": "light.set",
        "description": "Switch a light on or off",
        "parameters": {
            "room": {"type": "string", "description": "which room", "required": True},
            "state": {"type": "string", "enum": ["on", "off"], "required": True},
        },
    },
    "task.create": {
        "name": "task.create",
        "description": "Add a task to the todo list",
        "parameters": {
            "title": {"type": "string", "description": "what to do", "required": True},
            "due": {"type": "string", "format": "date", "description": "deadline"},
            "priority": {"type": "string", "enum": ["low", "normal", "high"]},
        },
    },
    "reminder.set": {
        "name": "reminder.set",
        "description": "Create a reminder message",
        "parameters": {
            "text": {"type": "string", "description": "what to be reminded of", "required": True},
            "repeat": {"type": "boolean", "description": "whether it repeats"},
        },
    },
}

HOLDOUT_TOOLS: dict[str, dict[str, Any]] = {
    "music.play": {
        "name": "music.play",
        "description": "Play music matching a query",
        "parameters": {
            "query": {"type": "string", "description": "what to play", "required": True},
        },
    },
    "note.create": {
        "name": "note.create",
        "description": "Write down a short note",
        "parameters": {
            "text": {"type": "string", "description": "content of the note", "required": True},
        },
    },
}

ALL_TOOLS = {**TRAIN_TOOLS, **HOLDOUT_TOOLS}
FAMILY_OF = {name: name.split(".")[0] for name in ALL_TOOLS}

# --------------------------------------------------------------------------
# Phrase banks
# --------------------------------------------------------------------------

# (surface, RELATIVE_DATETIME payload)
WHEN = {
    "en": [
        ("tomorrow afternoon", {"relation": "TOMORROW", "daypart": "AFTERNOON"}),
        ("tomorrow morning", {"relation": "TOMORROW", "daypart": "MORNING"}),
        ("today", {"relation": "TODAY"}),
        ("next friday", {"relation": "NEXT", "weekday": "FRIDAY"}),
        ("this monday evening", {"relation": "THIS", "weekday": "MONDAY", "daypart": "EVENING"}),
    ],
    "de": [
        ("morgen nachmittag", {"relation": "TOMORROW", "daypart": "AFTERNOON"}),
        ("morgen früh", {"relation": "TOMORROW", "daypart": "MORNING"}),
        ("heute", {"relation": "TODAY"}),
        ("nächsten freitag", {"relation": "NEXT", "weekday": "FRIDAY"}),
        ("diesen montagabend", {"relation": "THIS", "weekday": "MONDAY", "daypart": "EVENING"}),
    ],
    "fr": [
        ("demain après-midi", {"relation": "TOMORROW", "daypart": "AFTERNOON"}),
        ("demain matin", {"relation": "TOMORROW", "daypart": "MORNING"}),
        ("aujourd'hui", {"relation": "TODAY"}),
        ("vendredi prochain", {"relation": "NEXT", "weekday": "FRIDAY"}),
        ("lundi soir", {"relation": "THIS", "weekday": "MONDAY", "daypart": "EVENING"}),
    ],
    "es": [
        ("mañana por la tarde", {"relation": "TOMORROW", "daypart": "AFTERNOON"}),
        ("mañana por la mañana", {"relation": "TOMORROW", "daypart": "MORNING"}),
        ("hoy", {"relation": "TODAY"}),
        ("el próximo viernes", {"relation": "NEXT", "weekday": "FRIDAY"}),
        ("este lunes por la noche", {"relation": "THIS", "weekday": "MONDAY", "daypart": "EVENING"}),
    ],
}

# (surface, DATE payload) for task due dates
WHEN_DATE = {
    "en": [("tomorrow", {"relation": "TOMORROW"}), ("next friday", {"relation": "NEXT", "weekday": "FRIDAY"})],
    "de": [("morgen", {"relation": "TOMORROW"}), ("nächsten freitag", {"relation": "NEXT", "weekday": "FRIDAY"})],
    "fr": [("demain", {"relation": "TOMORROW"}), ("vendredi prochain", {"relation": "NEXT", "weekday": "FRIDAY"})],
    "es": [("mañana", {"relation": "TOMORROW"}), ("el próximo viernes", {"relation": "NEXT", "weekday": "FRIDAY"})],
}

# (surface, DURATION payload)
DUR = {
    "en": [
        ("one hour", {"magnitude": 1, "unit": "HOUR"}),
        ("90 minutes", {"magnitude": 90, "unit": "MINUTE"}),
        ("half an hour", {"magnitude": 0.5, "unit": "HOUR"}),
        ("ten minutes", {"magnitude": 10, "unit": "MINUTE"}),
        ("one and a half hours", {"magnitude": 1.5, "unit": "HOUR"}),
    ],
    "de": [
        ("eine stunde", {"magnitude": 1, "unit": "HOUR"}),
        ("90 minuten", {"magnitude": 90, "unit": "MINUTE"}),
        ("eine halbe stunde", {"magnitude": 0.5, "unit": "HOUR"}),
        ("zehn minuten", {"magnitude": 10, "unit": "MINUTE"}),
        ("eineinhalb stunden", {"magnitude": 1.5, "unit": "HOUR"}),
    ],
    "fr": [
        ("une heure", {"magnitude": 1, "unit": "HOUR"}),
        ("90 minutes", {"magnitude": 90, "unit": "MINUTE"}),
        ("une demi-heure", {"magnitude": 0.5, "unit": "HOUR"}),
        ("dix minutes", {"magnitude": 10, "unit": "MINUTE"}),
    ],
    "es": [
        ("una hora", {"magnitude": 1, "unit": "HOUR"}),
        ("90 minutos", {"magnitude": 90, "unit": "MINUTE"}),
        ("media hora", {"magnitude": 0.5, "unit": "HOUR"}),
        ("diez minutos", {"magnitude": 10, "unit": "MINUTE"}),
    ],
}

TITLES = {
    "en": ["dentist appointment", "team meeting", "project review", "call with anna"],
    "de": ["zahnarzttermin", "teammeeting", "projektbesprechung", "telefonat mit anna"],
    "fr": ["rendez-vous chez le dentiste", "réunion d'équipe", "revue de projet"],
    "es": ["cita con el dentista", "reunión de equipo", "revisión del proyecto"],
}

TASKS = {
    "en": ["buy groceries", "renew the passport", "water the plants", "prepare the slides"],
    "de": ["einkaufen gehen", "den pass verlängern", "die pflanzen gießen"],
    "fr": ["faire les courses", "arroser les plantes", "préparer les diapositives"],
    "es": ["hacer la compra", "regar las plantas", "preparar la presentación"],
}

RECIPIENTS = {
    "en": ["anna müller", "tom", "the sales team", "dr. garcía"],
    "de": ["anna müller", "tom", "das vertriebsteam", "dr. garcía"],
    "fr": ["anna müller", "tom", "l'équipe commerciale"],
    "es": ["anna müller", "tom", "el equipo de ventas"],
}

SUBJECTS = {
    "en": ["the quarterly report", "the budget", "tomorrow's demo"],
    "de": ["den quartalsbericht", "das budget", "die demo von morgen"],
    "fr": ["le rapport trimestriel", "le budget"],
    "es": ["el informe trimestral", "el presupuesto"],
}

CITIES = {
    "en": ["berlin", "munich", "paris", "madrid"],
    "de": ["berlin", "münchen", "köln", "hamburg"],
    "fr": ["paris", "lyon", "marseille"],
    "es": ["madrid", "barcelona", "sevilla"],
}

ROOMS = {
    "en": ["kitchen", "bedroom", "living room", "office"],
    "de": ["küche", "schlafzimmer", "wohnzimmer", "büro"],
    "fr": ["cuisine", "chambre", "salon"],
    "es": ["cocina", "dormitorio", "salón"],
}

QUERIES = {
    "en": ["some jazz", "my focus playlist", "the new taylor swift album"],
    "de": ["etwas jazz", "meine fokus-playlist"],
    "fr": ["du jazz", "ma playlist de concentration"],
    "es": ["algo de jazz", "mi lista de reproducción"],
}

NOTES = {
    "en": ["the wifi password is sunflower42", "call the plumber on monday"],
    "de": ["das wlan-passwort ist sonnenblume42"],
    "fr": ["le code du garage est 4812"],
    "es": ["la contraseña del wifi es girasol42"],
}

NO_CALL_UTTERANCES = {
    "en": [
        "what does the timer tool do?",
        "did i already email anna the report?",
        "order a pizza for me",
        "thanks, that's all for now",
        "how does the calendar integration work?",
    ],
    "de": [
        "was macht das timer-tool?",
        "habe ich anna den bericht schon geschickt?",
        "bestell mir eine pizza",
        "danke, das ist alles",
    ],
    "fr": [
        "à quoi sert l'outil minuteur ?",
        "commande-moi une pizza",
        "merci, c'est tout",
    ],
    "es": [
        "¿qué hace la herramienta de temporizador?",
        "pídeme una pizza",
        "gracias, eso es todo",
    ],
}

LANGS = ["en", "de", "fr", "es"]


# --------------------------------------------------------------------------
# Utterance composition with span tracking
# --------------------------------------------------------------------------


def compose(*parts: Part) -> tuple[str, dict[str, tuple[int, int]]]:
    text = ""
    spans: dict[str, tuple[int, int]] = {}
    for p in parts:
        if isinstance(p, tuple):
            s = len(text)
            text += p[0]
            spans[p[1]] = (s, len(text))
        else:
            text += p
    return text, spans


def arg_entry(
    parameter: str,
    semantic_type: str,
    value: Any,
    spans: dict[str, tuple[int, int]],
    key: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "parameter": parameter,
        "semantic_type": semantic_type,
        "value": value,
    }
    if key is not None and key in spans:
        s, e = spans[key]
        entry["char_span"] = {"start": s, "end": e}
    return entry


# Per-domain CALL builders: (lang, rng) -> (utterance, arguments)
def build_calendar(lang: str, rng: random.Random) -> tuple[str, list[dict]]:
    title = rng.choice(TITLES[lang])
    when, when_v = rng.choice(WHEN[lang])
    with_dur = rng.random() < 0.5
    parts: list[Part] = {
        "en": ["schedule a ", (title, "title"), " ", (when, "start")],
        "de": ["plane einen ", (title, "title"), " ", (when, "start")],
        "fr": ["planifie un ", (title, "title"), " ", (when, "start")],
        "es": ["programa una ", (title, "title"), " ", (when, "start")],
    }[lang]
    args = []
    if with_dur:
        dur, dur_v = rng.choice(DUR[lang])
        joiner = {"en": " for ", "de": " für ", "fr": " pendant ", "es": " durante "}[lang]
        parts += [joiner, (dur, "duration_minutes")]
    text, spans = compose(*parts)
    args.append(arg_entry("title", "STRING", title, spans, "title"))
    args.append(arg_entry("start", "RELATIVE_DATETIME", when_v, spans, "start"))
    if with_dur:
        args.append(arg_entry("duration_minutes", "DURATION", dur_v, spans, "duration_minutes"))
    return text, args


def build_email(lang: str, rng: random.Random) -> tuple[str, list[dict]]:
    rcpt = rng.choice(RECIPIENTS[lang])
    with_subject = rng.random() < 0.6
    parts: list[Part] = {
        "en": ["send an email to ", (rcpt, "recipient")],
        "de": ["schick eine e-mail an ", (rcpt, "recipient")],
        "fr": ["envoie un e-mail à ", (rcpt, "recipient")],
        "es": ["envía un correo a ", (rcpt, "recipient")],
    }[lang]
    args = []
    if with_subject:
        subj = rng.choice(SUBJECTS[lang])
        joiner = {"en": " about ", "de": " über ", "fr": " au sujet de ", "es": " sobre "}[lang]
        parts += [joiner, (subj, "subject")]
    text, spans = compose(*parts)
    args.append(arg_entry("recipient", "STRING", rcpt, spans, "recipient"))
    if with_subject:
        args.append(arg_entry("subject", "STRING", subj, spans, "subject"))
    return text, args


def build_timer(lang: str, rng: random.Random) -> tuple[str, list[dict]]:
    dur, dur_v = rng.choice(DUR[lang])
    parts: list[Part] = {
        "en": ["set a timer for ", (dur, "length")],
        "de": ["stell einen timer auf ", (dur, "length")],
        "fr": ["mets un minuteur de ", (dur, "length")],
        "es": ["pon un temporizador de ", (dur, "length")],
    }[lang]
    text, spans = compose(*parts)
    return text, [arg_entry("length", "DURATION", dur_v, spans, "length")]


def build_weather(lang: str, rng: random.Random) -> tuple[str, list[dict]]:
    city = rng.choice(CITIES[lang])
    with_days = rng.random() < 0.4
    parts: list[Part] = {
        "en": ["what's the weather in ", (city, "city")],
        "de": ["wie wird das wetter in ", (city, "city")],
        "fr": ["quel temps fera-t-il à ", (city, "city")],
        "es": ["qué tiempo hará en ", (city, "city")],
    }[lang]
    args = []
    if with_days:
        days = rng.choice(["2", "3", "5"])
        pre, post = {
            "en": (" for the next ", " days"),
            "de": (" für die nächsten ", " tage"),
            "fr": (" pour les ", " prochains jours"),
            "es": (" para los próximos ", " días"),
        }[lang]
        parts += [pre, (days, "days"), post]
    text, spans = compose(*parts)
    args.insert(0, arg_entry("city", "LOCATION", {"text": city}, spans, "city"))
    if with_days:
        args.append(arg_entry("days", "INTEGER", int(days), spans, "days"))
    return text, args


def build_light(lang: str, rng: random.Random) -> tuple[str, list[dict]]:
    room = rng.choice(ROOMS[lang])
    on = rng.random() < 0.5
    # Several verb frames per language (schalten/machen/drehen, turn/switch,
    # allumer/éteindre/couper …) so the surface form generalizes.
    variants: list[list[Part]]
    if lang == "en":
        surface = "on" if on else "off"
        variants = [
            ["turn ", (surface, "state"), " the light in the ", (room, "room")],
            ["switch ", (surface, "state"), " the light in the ", (room, "room")],
            ["turn the light in the ", (room, "room"), " ", (surface, "state")],
        ]
    elif lang == "de":
        variants = [
            ["schalte das licht im ", (room, "room"), " ", ("ein" if on else "aus", "state")],
            ["mach das licht im ", (room, "room"), " ", ("an" if on else "aus", "state")],
            ["drehe das licht im ", (room, "room"), " ", ("auf" if on else "ab", "state")],
            ["dreh das licht im ", (room, "room"), " ", ("auf" if on else "ab", "state")],
        ]
    elif lang == "fr":
        variants = [
            [("allume" if on else "éteins", "state"), " la lumière dans la ", (room, "room")],
            [("allume" if on else "coupe", "state"), " la lumière dans la ", (room, "room")],
        ]
    else:
        variants = [
            [("enciende" if on else "apaga", "state"), " la luz del ", (room, "room")],
        ]
    parts = rng.choice(variants)
    text, spans = compose(*parts)
    enum_value = {"index": 0 if on else 1, "symbol": "on" if on else "off"}
    return text, [
        arg_entry("room", "STRING", room, spans, "room"),
        arg_entry("state", "ENUM", enum_value, spans, "state"),
    ]


def build_task(lang: str, rng: random.Random) -> tuple[str, list[dict]]:
    task = rng.choice(TASKS[lang])
    with_due = rng.random() < 0.5
    with_prio = rng.random() < 0.5
    parts: list[Part] = {
        "en": ["add a task to ", (task, "title")],
        "de": ["erstelle eine aufgabe: ", (task, "title")],
        "fr": ["ajoute une tâche : ", (task, "title")],
        "es": ["añade una tarea: ", (task, "title")],
    }[lang]
    args = [None]  # placeholder for ordered insert
    if with_due:
        due, due_v = rng.choice(WHEN_DATE[lang])
        joiner = {"en": " due ", "de": " fällig ", "fr": " pour ", "es": " para "}[lang]
        parts += [joiner, (due, "due")]
    if with_prio:
        prio_idx = rng.choice([0, 2])
        prio_surface = {
            "en": ["low", "normal", "high"],
            "de": ["niedriger", "normaler", "hoher"],
            "fr": ["basse", "normale", "haute"],
            "es": ["baja", "normal", "alta"],
        }[lang][prio_idx]
        pre, post = {
            "en": (" with ", " priority"),
            "de": (" mit ", " priorität"),
            "fr": (" avec une priorité ", ""),
            "es": (" con prioridad ", ""),
        }[lang]
        parts += [pre, (prio_surface, "priority")]
        if post:
            parts += [post]
    text, spans = compose(*parts)
    args[0] = arg_entry("title", "STRING", task, spans, "title")
    if with_due:
        args.append(arg_entry("due", "RELATIVE_DATE", due_v, spans, "due"))
    if with_prio:
        symbol = ["low", "normal", "high"][prio_idx]
        args.append(
            arg_entry("priority", "ENUM", {"index": prio_idx, "symbol": symbol}, spans, "priority")
        )
    return text, args


def build_reminder(lang: str, rng: random.Random) -> tuple[str, list[dict]]:
    note = rng.choice(TASKS[lang])
    repeating = rng.random() < 0.5
    rep_surface = {
        "en": "repeating ",
        "de": "wiederkehrende ",
        "fr": "récurrent ",
        "es": "recurrente ",
    }[lang]
    parts: list[Part]
    if lang == "en":
        parts = ["set a ", *([(rep_surface, "repeat")] if repeating else []), "reminder to ", (note, "text")]
    elif lang == "de":
        parts = ["erstelle eine ", *([(rep_surface, "repeat")] if repeating else []), "erinnerung: ", (note, "text")]
    elif lang == "fr":
        parts = ["crée un rappel ", *([(rep_surface, "repeat")] if repeating else []), ": ", (note, "text")]
    else:
        parts = ["crea un recordatorio ", *([(rep_surface, "repeat")] if repeating else []), ": ", (note, "text")]
    text, spans = compose(*parts)
    args = [arg_entry("text", "STRING", note, spans, "text")]
    if repeating:
        args.append(arg_entry("repeat", "BOOLEAN", True, spans, "repeat"))
    return text, args


def build_music(lang: str, rng: random.Random) -> tuple[str, list[dict]]:
    q = rng.choice(QUERIES[lang])
    parts: list[Part] = {
        "en": ["play ", (q, "query")],
        "de": ["spiel ", (q, "query")],
        "fr": ["joue ", (q, "query")],
        "es": ["pon ", (q, "query")],
    }[lang]
    text, spans = compose(*parts)
    return text, [arg_entry("query", "STRING", q, spans, "query")]


def build_note(lang: str, rng: random.Random) -> tuple[str, list[dict]]:
    n = rng.choice(NOTES[lang])
    parts: list[Part] = {
        "en": ["take a note: ", (n, "text")],
        "de": ["notiere: ", (n, "text")],
        "fr": ["prends une note : ", (n, "text")],
        "es": ["toma nota: ", (n, "text")],
    }[lang]
    text, spans = compose(*parts)
    return text, [arg_entry("text", "STRING", n, spans, "text")]


CALL_BUILDERS = {
    "calendar.create": build_calendar,
    "email.send": build_email,
    "timer.set": build_timer,
    "weather.lookup": build_weather,
    "light.set": build_light,
    "task.create": build_task,
    "reminder.set": build_reminder,
    "music.play": build_music,
    "note.create": build_note,
}

# ASK builders: utterance missing a required arg → (utterance, missing_param)
ASK_BUILDERS: dict[str, dict[str, str]] = {
    "calendar.create": {
        "en": "schedule a dentist appointment|start",
        "de": "plane einen zahnarzttermin|start",
        "fr": "planifie un rendez-vous chez le dentiste|start",
        "es": "programa una cita con el dentista|start",
    },
    "email.send": {
        "en": "send an email about the budget|recipient",
        "de": "schick eine e-mail über das budget|recipient",
        "fr": "envoie un e-mail au sujet du budget|recipient",
        "es": "envía un correo sobre el presupuesto|recipient",
    },
    "timer.set": {
        "en": "set a timer|length",
        "de": "stell einen timer|length",
        "fr": "mets un minuteur|length",
        "es": "pon un temporizador|length",
    },
    "light.set": {
        "en": "turn on the light|room",
        "de": "mach das licht an|room",
        "fr": "allume la lumière|room",
        "es": "enciende la luz|room",
    },
}


# --------------------------------------------------------------------------
# Mutations
# --------------------------------------------------------------------------


def randomize_names(example: dict, rng: random.Random) -> dict:
    """Spec §49: rename tools/args to opaque ids in candidates AND gold."""
    ex = json.loads(json.dumps(example))
    for tool in ex["candidates"]:
        old_tool = tool["name"]
        new_tool = f"fn_{rng.randrange(100, 999)}"
        arg_map = {}
        new_params = {}
        for k, (pname, pdef) in enumerate(tool.get("parameters", {}).items()):
            new_name = f"arg_{k}"
            arg_map[pname] = new_name
            new_params[new_name] = pdef
        tool["parameters"] = new_params
        tool["name"] = new_tool
        if ex["gold"].get("tool") == old_tool:
            ex["gold"]["tool"] = new_tool
            for a in ex["gold"].get("arguments", []):
                a["parameter"] = arg_map[a["parameter"]]
            for u in ex["gold"].get("unresolved", []):
                u["parameter"] = arg_map[u["parameter"]]
    return ex


# --------------------------------------------------------------------------
# Example assembly
# --------------------------------------------------------------------------


def make_id(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def assemble(
    rng: random.Random,
    lang: str,
    gold_tool: str | None,
    utterance: str,
    arguments: list[dict],
    unresolved: list[dict],
    action: str,
    split: str,
    tags: list[str],
) -> dict:
    decoy_pool = [t for t in TRAIN_TOOLS if t != gold_tool]
    n_decoys = rng.randint(1, 3)
    decoys = rng.sample(decoy_pool, n_decoys)
    candidates = [ALL_TOOLS[t] for t in ([gold_tool] if gold_tool else []) + decoys]
    rng.shuffle(candidates)
    ex = {
        "id": "",
        "lang": lang,
        "utterance": utterance,
        "candidates": candidates,
        "gold": {
            "action": action,
            "tool": gold_tool,
            "arguments": arguments,
            "unresolved": unresolved,
        },
        "split": split,
        "tags": tags,
    }
    ex["id"] = make_id(ex)
    return ex


def punctuate(rng: random.Random, utterance: str) -> str:
    r = rng.random()
    if r < 0.25:
        return utterance + "!"
    if r < 0.4:
        return utterance + "."
    return utterance


def generate(seed: int = 7, per_domain_lang: int = 30) -> list[dict]:
    rng = random.Random(seed)
    examples: list[dict] = []

    # CALL examples for train-pool families.
    for tool_name in TRAIN_TOOLS:
        builder = CALL_BUILDERS[tool_name]
        for lang in LANGS:
            for i in range(per_domain_lang):
                split = "train" if i < per_domain_lang - 6 else ("dev" if i % 2 == 0 else "test")
                utterance, args = builder(lang, rng)
                utterance = punctuate(rng, utterance)
                examples.append(
                    assemble(rng, lang, tool_name, utterance, args, [], "CALL", split, ["call"])
                )

    # ASK examples.
    for tool_name, per_lang in ASK_BUILDERS.items():
        for lang in LANGS:
            for i in range(per_domain_lang // 3):
                utterance, missing = per_lang[lang].split("|")
                split = "train" if i < per_domain_lang // 3 - 2 else ("dev" if i % 2 == 0 else "test")
                examples.append(
                    assemble(
                        rng,
                        lang,
                        tool_name,
                        utterance,
                        [],
                        [{"parameter": missing, "reason": "MISSING"}],
                        "ASK",
                        split,
                        ["ask"],
                    )
                )

    # NO_CALL examples.
    for lang in LANGS:
        for i in range(per_domain_lang):
            utterance = rng.choice(NO_CALL_UTTERANCES[lang])
            split = "train" if i < per_domain_lang - 6 else ("dev" if i % 2 == 0 else "test")
            examples.append(
                assemble(rng, lang, None, utterance, [], [], "NO_CALL", split, ["no_call"])
            )

    # Unseen-family eval examples (test only).
    for tool_name in HOLDOUT_TOOLS:
        builder = CALL_BUILDERS[tool_name]
        for lang in LANGS:
            for _ in range(8):
                utterance, args = builder(lang, rng)
                examples.append(
                    assemble(
                        rng, lang, tool_name, utterance, args, [], "CALL", "test",
                        ["call", "unseen_family"],
                    )
                )

    # Name-randomization: augmentation on train, dedicated masked eval split.
    augmented = []
    for ex in examples:
        if ex["split"] == "train" and rng.random() < 0.25:
            augmented.append({**randomize_names(ex, rng), "tags": [*ex["tags"], "randomized_names"]})
        if ex["split"] == "test" and "unseen_family" not in ex["tags"] and rng.random() < 0.35:
            masked = randomize_names(ex, rng)
            masked["tags"] = [*ex["tags"], "masked_names"]
            masked["id"] = make_id(masked)
            augmented.append(masked)
    for ex in augmented:
        ex["id"] = make_id(ex)
    examples.extend(augmented)

    # Validate everything (raises on any construction bug), then dedup by id.
    seen: dict[str, dict] = {}
    for ex in examples:
        DatasetExample.model_validate(ex)
        seen.setdefault(ex["id"], ex)
    return list(seen.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/mini"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--per-domain-lang", type=int, default=30)
    args = parser.parse_args()

    examples = generate(args.seed, args.per_domain_lang)
    args.out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    files = {}
    for split in ("train", "dev", "test"):
        files[split] = (args.out / f"{split}.jsonl").open("w")
    for ex in examples:
        files[ex["split"]].write(json.dumps(ex, ensure_ascii=False) + "\n")
        counts[ex["split"]] = counts.get(ex["split"], 0) + 1
        for tag in ex["tags"]:
            counts[f"tag:{tag}"] = counts.get(f"tag:{tag}", 0) + 1
        counts[f"lang:{ex['lang']}"] = counts.get(f"lang:{ex['lang']}", 0) + 1
    for f in files.values():
        f.close()
    (args.out / "stats.json").write_text(json.dumps(counts, indent=2, sort_keys=True))
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
