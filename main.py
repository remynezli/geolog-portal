from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import re
from difflib import SequenceMatcher


BASE_DIR = Path(__file__).resolve().parent
JSON_PATH = BASE_DIR / "Challenges.json"

app = FastAPI()

# CORS (useful if you ever serve frontend separately)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- DATA LOADING -----------------
def fuzzy_match(a: str, b: str, threshold: int = 80) -> bool:
    if not a or not b:
        return False
    ratio = SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100
    return ratio >= threshold

def load_challenges(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for row in data:
        lvl1 = row.get("level_1_solution", [])
        lvl2 = row.get("level_2_solution", [])

        if not isinstance(lvl1, list):
            lvl1 = [lvl1] if lvl1 is not None else []
        if not isinstance(lvl2, list):
            lvl2 = [lvl2] if lvl2 is not None else []

        seen = set()
        solutions = []
        for s in lvl1 + lvl2:
            if s and s not in seen:
                seen.add(s)
                solutions.append(s)

        row["solutions"] = solutions

        tags = row.get("tags", [])
        if not isinstance(tags, list):
            tags = [tags] if tags is not None else []
        row["tags"] = tags

        row.setdefault("macrotopic", "")
        row.setdefault("topic", "")
        row.setdefault("challenge", "")

    return data

def save_challenges(path: Path, data: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

data: List[Dict[str, Any]] = load_challenges(JSON_PATH)
class NewChallenge(BaseModel):
    macrotopic: str
    topic: str
    challenge: str
    tags: List[str] = []
    level_1_solution: List[str] = []
    level_2_solution: List[str] = []

# ----------------- SEARCH HELPERS -----------------

SYNONYM_GROUPS = {
    "losses": ["losses", "lost circulation", "lost circ", "mud loss", "lcm"],
    "vibration": [
        "vibration", "vibrations", "stick slip", "stick-slip",
        "torsion", "torsional", "torque", "shock"
    ],
    "instability": [
        "instability", "borehole instability", "cavings",
        "caving", "breakout", "collapse", "wellbore collapse"
    ],
    "hpht": ["hpht", "high pressure", "high temp", "high temperature"],
    "fracture": ["fracture", "fractured", "fault", "faults"],
    "gas": ["gas", "gas peaks", "gas spike", "kick"],
    "bit wear": ["bit wear", "worn bit", "bit damage"],
    "mineralogy": [
        "mineralogy", "mineralogical", "mineralogical analysis",
        "xrd", "xrf"
    ],
}

def expand_keywords(words: List[str]) -> List[str]:
    expanded = set()
    for w in words:
        w = w.lower()
        expanded.add(w)
        for group_words in SYNONYM_GROUPS.values():
            if any(w == gw for gw in group_words):
                expanded.update(group_words)
    return list(expanded)

def keyword_matches_text(keyword: str, text: str) -> bool:
    keyword = keyword.lower()
    text = text.lower()
    if keyword in text:
        return True
    tokens = [t for t in re.split(r"\W+", text) if len(t) >= 3]
    for token in tokens:
        if keyword in token or token in keyword:
            return True
    return False

def filter_challenges(
    macrotopic: Optional[str],
    topic: Optional[str],
    selected_tags: List[str],
    search_text: str,
) -> List[Dict[str, Any]]:
    results = data

    if macrotopic:
        results = [
            r for r in results
            if r.get("macrotopic", "").lower() == macrotopic.lower()
        ]

    if topic:
        results = [
            r for r in results
            if r.get("topic", "").lower() == topic.lower()
        ]

    if selected_tags:
        tags_lower = {t.lower() for t in selected_tags}
        results = [
            r for r in results
            if tags_lower.intersection({t.lower() for t in r.get("tags", [])})
        ]

    search_text = (search_text or "").strip().lower()

    # 🔹 If no free-text query: just return the filtered list (by macro/topic/tags)
    if not search_text:
        return results

    words = [w for w in re.split(r"\W+", search_text) if len(w) >= 3]
    if not words:
        return results

    keywords = [k.lower() for k in expand_keywords(words)]

    scored: List[tuple[int, Dict[str, Any]]] = []
    for r in results:
        challenge_text = r.get("challenge", "") or ""
        topic_text = r.get("topic", "") or ""
        tags_list = r.get("tags", []) or []
        tags_text = " ".join(tags_list)

        score = 0
        for kw in keywords:
            # TAGS scoring (highest weight)
            if keyword_matches_text(kw, tags_text) or fuzzy_match(kw, tags_text):
                score += 3

            # TOPIC scoring
            if keyword_matches_text(kw, topic_text) or fuzzy_match(kw, topic_text):
                score += 2

            # CHALLENGE TEXT scoring
            if keyword_matches_text(kw, challenge_text) or fuzzy_match(kw, challenge_text):
                score += 1

        scored.append((score, r))

    # ⛔️ HERE IS THE IMPORTANT PART
    # Only keep items with score > 0 when there is a query
    scored = [(s, r) for s, r in scored if s > 0]

    # Optional: if nothing matched, return empty list
    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for score, r in scored]

# ----------------- API MODELS -----------------

class SearchRequest(BaseModel):
    macrotopic: Optional[str] = None
    topic: Optional[str] = None
    tags: List[str] = []
    query: str = ""


# ----------------- API ENDPOINTS -----------------

@app.get("/challenges")
def list_challenges():
    return data

@app.post("/search")
def search(request: SearchRequest):
    results = filter_challenges(
        macrotopic=request.macrotopic,
        topic=request.topic,
        selected_tags=request.tags,
        search_text=request.query,
    )
    return results
@app.post("/add-challenge")
def add_challenge(payload: NewChallenge):
    row = payload.model_dump()

    # Normalize solutions (same style as load_challenges)
    lvl1 = row.get("level_1_solution", [])
    lvl2 = row.get("level_2_solution", [])

    if not isinstance(lvl1, list):
        lvl1 = [lvl1] if lvl1 is not None else []
    if not isinstance(lvl2, list):
        lvl2 = [lvl2] if lvl2 is not None else []

    seen = set()
    solutions = []
    for s in lvl1 + lvl2:
        if s and s not in seen:
            seen.add(s)
            solutions.append(s)

    row["level_1_solution"] = lvl1
    row["level_2_solution"] = lvl2
    row["solutions"] = solutions

    tags = row.get("tags", [])
    if not isinstance(tags, list):
        tags = [tags] if tags is not None else []
    row["tags"] = tags

    row.setdefault("macrotopic", "")
    row.setdefault("topic", "")
    row.setdefault("challenge", "")

    # append in memory + save
    data.append(row)
    save_challenges(JSON_PATH, data)

    return {"status": "ok", "challenge": row}
