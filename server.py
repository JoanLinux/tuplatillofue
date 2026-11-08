"""
TuPlatilloFue — Restaurant face recognition display
Accede en: http://localhost:5151
Pantalla TV: http://localhost:5151/pantalla
Admin:       http://localhost:5151/admin
"""

import os
# Point InsightFace to bundled models — no internet required
os.environ["INSIGHTFACE_HOME"] = os.path.join(os.path.dirname(__file__), "models", "insightface")

import cv2
import numpy as np
import threading
import queue
import time
import json
import sqlite3
from pathlib import Path
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from waitress import serve

app = Flask(__name__, static_folder="static")
CORS(app)

BASE         = Path(__file__).parent
DB_PATH      = BASE / "tuplatillo.db"
FACES_DIR    = BASE / "uploads" / "faces"
DISHES_DIR   = BASE / "uploads" / "dishes"
VIDEOS_DIR   = BASE / "uploads" / "videos"
BRAND_DIR    = BASE / "uploads" / "brand"
PLAYLIST_DIR = BASE / "uploads" / "playlist"
for d in (FACES_DIR, DISHES_DIR, VIDEOS_DIR, BRAND_DIR, PLAYLIST_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── STATE ──────────────────────────────────────────────────────────────────────
state = {
    "running":        False,
    "current":        None,
    "fps":            0.0,
    "error":          "",
    "source":         "0",
    "loop":           True,
    "playlist_on":    False,
    "playlist_version": 0,
}
frame_lock    = threading.Lock()
latest_frame  = None
stream_thread = None
YOLO_MODEL    = None
FACE_APP      = None   # InsightFace

# ── DATABASE ───────────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            face_path TEXT NOT NULL,
            embedding TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS dishes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            image_path TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER REFERENCES customers(id),
            dish_id     INTEGER REFERENCES dishes(id),
            ordered_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS playlist (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            type      TEXT NOT NULL CHECK(type IN ('image','video','horoscope')),
            filename  TEXT NOT NULL DEFAULT '',
            label     TEXT,
            show_text INTEGER NOT NULL DEFAULT 1,
            duration  INTEGER NOT NULL DEFAULT 6,
            position  INTEGER NOT NULL DEFAULT 0,
            signs     TEXT
        );
        CREATE TABLE IF NOT EXISTS horoscopes (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            sign    TEXT NOT NULL,
            text    TEXT NOT NULL DEFAULT "",
            date    TEXT NOT NULL,
            source  TEXT DEFAULT "manual",
            UNIQUE(sign, date)
        );
        CREATE TABLE IF NOT EXISTS horoscope_sources (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            url        TEXT NOT NULL,
            active     INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (date('now'))
        );
    """)
    con.commit()
    con.close()

def get_db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def last_dish_for(customer_id):
    con = get_db()
    row = con.execute("""
        SELECT d.name, d.description, d.image_path
        FROM orders o JOIN dishes d ON d.id = o.dish_id
        WHERE o.customer_id = ?
        ORDER BY o.ordered_at DESC LIMIT 1
    """, (customer_id,)).fetchone()
    con.close()
    return dict(row) if row else None


# ── MODELS ─────────────────────────────────────────────────────────────────────
def get_yolo():
    global YOLO_MODEL
    if YOLO_MODEL is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Cargando YOLOv8n en {device}...")
        YOLO_MODEL = YOLO("yolov8n.pt")
        YOLO_MODEL.to(device)
        print(f"[INFO] YOLO listo en {device}.")
    return YOLO_MODEL

def get_face_app():
    global FACE_APP
    if FACE_APP is None:
        import torch
        from insightface.app import FaceAnalysis
        has_cuda = torch.cuda.is_available()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if has_cuda else ["CPUExecutionProvider"]
        print(f"[INFO] Cargando InsightFace buffalo_sc en {'CUDA' if has_cuda else 'CPU'}...")
        FACE_APP = FaceAnalysis(name="buffalo_sc", providers=providers)
        FACE_APP.prepare(ctx_id=0 if has_cuda else -1, det_size=(320, 320))
        print("[INFO] InsightFace listo.")
    return FACE_APP


# ── FACE HELPERS ───────────────────────────────────────────────────────────────
def cosine_sim(a, b):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def compute_embedding(img_bgr):
    """InsightFace embedding from BGR image. Returns list[float] or None."""
    try:
        fa = get_face_app()
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        faces = fa.get(rgb)
        if not faces:
            return None
        # largest face by bbox area
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        return face.embedding.tolist()
    except Exception as e:
        print(f"[EMBED] {e}")
        return None

def detect_faces_in_image(img_bgr):
    """Returns list of {x,y,w,h} for each detected face."""
    try:
        fa = get_face_app()
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        faces = fa.get(rgb)
        result = []
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            result.append({"x": x1, "y": y1, "w": x2-x1, "h": y2-y1,
                           "img_w": img_bgr.shape[1], "img_h": img_bgr.shape[0]})
        return result
    except Exception as e:
        print(f"[DETECT] {e}")
        return []

def find_customer(face_bgr, threshold=0.35):
    """Match face crop against stored customers. Returns match dict or None."""
    emb = compute_embedding(face_bgr)
    if emb is None:
        return None
    con = get_db()
    customers = con.execute(
        "SELECT id, name, embedding FROM customers WHERE embedding IS NOT NULL"
    ).fetchall()
    con.close()
    best_id, best_name, best_sim = None, None, -1
    for c in customers:
        try:
            stored = json.loads(c["embedding"])
            sim = cosine_sim(emb, stored)
            if sim > best_sim:
                best_sim, best_id, best_name = sim, c["id"], c["name"]
        except Exception:
            continue
    if best_sim >= threshold:
        return {"id": best_id, "name": best_name, "sim": best_sim}
    return None


# ── FACE WORKER THREAD ─────────────────────────────────────────────────────────
# Stream thread puts face crops here; worker processes without blocking stream.
face_queue = queue.Queue(maxsize=2)   # small buffer — drop old if full

def face_worker():
    while True:
        try:
            crop = face_queue.get(timeout=1)
        except queue.Empty:
            continue
        if crop is None:
            break
        match = find_customer(crop)
        if match:
            dish = last_dish_for(match["id"])
            con2 = get_db()
            cust_row = con2.execute("SELECT face_path FROM customers WHERE id=?", (match["id"],)).fetchone()
            con2.close()
            face_file = None
            if cust_row and cust_row["face_path"]:
                face_file = Path(cust_row["face_path"]).name
            state["current"] = {
                "customer_name": match["name"],
                "customer_face": face_file,
                "dish_name":     dish["name"]        if dish else "Sin pedido previo",
                "dish_desc":     dish["description"] if dish else "",
                "dish_image":    dish["image_path"]  if dish else None,
                "detected_at":   time.time(),
            }

face_worker_thread = threading.Thread(target=face_worker, daemon=True)
face_worker_thread.start()


# ── STREAM THREAD ─────────────────────────────────────────────────────────────
YOLO_EVERY      = 1     # run YOLO every frame (CUDA is fast)
DETECT_INTERVAL = 10    # enqueue face crop every N YOLO frames
MAX_W           = 640

def run_stream():
    global latest_frame, state

    url = state["source"]
    try:
        url = int(url)
    except ValueError:
        pass

    model = get_yolo()
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        state["error"] = f"No se pudo abrir: {url}"
        state["running"] = False
        return

    if isinstance(url, str):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    state["error"] = ""

    import supervision as sv
    box_ann = sv.BoxAnnotator(thickness=2)
    last_detections = None
    raw_n = 0
    proc_n = 0
    fps_counter, fps_start = 0, time.time()

    while state["running"]:
        ret, frame = cap.read()
        if not ret:
            if isinstance(url, str) and os.path.isfile(url):
                if state.get("loop", True):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            time.sleep(0.03)
            continue

        raw_n += 1

        # Resize for YOLO
        h, w = frame.shape[:2]
        if w > MAX_W:
            scale = MAX_W / w
            frame = cv2.resize(frame, (MAX_W, int(h * scale)), interpolation=cv2.INTER_LINEAR)

        display = frame

        if raw_n % YOLO_EVERY == 0:
            proc_n += 1
            results = model.track(
                frame, conf=0.45, classes=[0],
                persist=True, tracker="bytetrack.yaml",
                imgsz=416, max_det=15, verbose=False
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            last_detections = detections

            # Enqueue face crop for recognition (non-blocking)
            if proc_n % DETECT_INTERVAL == 0 and len(detections) > 0:
                fh, fw = frame.shape[:2]
                for box in detections.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    face_h = int((y2 - y1) * 0.5)
                    fx1 = max(0, x1); fy1 = max(0, y1)
                    fx2 = min(fw, x2); fy2 = min(fh, y1 + face_h)
                    if fx2 > fx1 and fy2 > fy1:
                        crop = frame[fy1:fy2, fx1:fx2].copy()
                        try:
                            face_queue.put_nowait(crop)
                        except queue.Full:
                            pass  # skip — worker busy, don't block
                        break

        # Annotate
        if last_detections is not None:
            display = box_ann.annotate(frame.copy(), last_detections)

        # Clear stale detection
        if state["current"] and time.time() - state["current"]["detected_at"] > 8:
            state["current"] = None

        # FPS
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            state["fps"] = round(fps_counter / elapsed, 1)
            fps_counter = 0
            fps_start = time.time()

        # Encode every frame — CUDA YOLO is fast enough
        _, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with frame_lock:
            latest_frame = buf.tobytes()

    cap.release()
    state["running"] = False


# ── RUTAS ─────────────────────────────────────────────────────────────────────

@app.route("/")
@app.route("/admin")
def admin():
    return send_from_directory("static", "admin.html")

@app.route("/pantalla")
def pantalla():
    return send_from_directory("static", "pantalla.html")

@app.route("/frame")
def single_frame():
    with frame_lock:
        frame = latest_frame
    if frame is None:
        return Response(status=204)
    return Response(frame, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})

@app.route("/stream")
def mjpeg_stream():
    def generate():
        while True:
            with frame_lock:
                frame = latest_frame
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.033)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

@app.route("/status")
def status():
    alive = stream_thread is not None and stream_thread.is_alive()
    if state["running"] and not alive:
        state["running"] = False
    return jsonify({
        "running":          state["running"],
        "fps":              state["fps"],
        "error":            state["error"],
        "current":          state["current"],
        "playlist_on":      state["playlist_on"],
        "playlist_version": state["playlist_version"],
    })

@app.route("/start", methods=["POST"])
def start():
    global stream_thread
    if state["running"] and (stream_thread is None or not stream_thread.is_alive()):
        state["running"] = False
    if state["running"]:
        return jsonify({"ok": False, "msg": "Ya corriendo"})
    data = request.json or {}
    state["source"]  = data.get("source", "0")
    state["loop"]    = bool(data.get("loop", True))
    state["running"] = True
    stream_thread = threading.Thread(target=run_stream, daemon=True)
    stream_thread.start()
    return jsonify({"ok": True})

@app.route("/stop", methods=["POST"])
def stop():
    state["running"] = False
    state["current"] = None   # clear so pantalla returns to playlist immediately
    return jsonify({"ok": True})


# ── PLAYLIST API ───────────────────────────────────────────────────────────────

@app.route("/api/playlist", methods=["GET"])
def list_playlist():
    con = get_db()
    rows = con.execute("SELECT * FROM playlist ORDER BY position, id").fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/playlist/upload", methods=["POST"])
def upload_playlist_item():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]
    fname = secure_filename(f.filename)
    ext = Path(fname).suffix.lower()
    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    vid_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".ts"}
    if ext in img_exts:
        ftype = "image"
    elif ext in vid_exts:
        ftype = "video"
    else:
        return jsonify({"error": "formato no soportado"}), 400
    path = PLAYLIST_DIR / fname
    f.save(str(path))
    label    = request.form.get("label", Path(fname).stem)
    duration = int(request.form.get("duration", 6))
    con = get_db()
    max_pos = con.execute("SELECT COALESCE(MAX(position),0)+1 FROM playlist").fetchone()[0]
    cur = con.execute(
        "INSERT INTO playlist (type,filename,label,duration,position) VALUES (?,?,?,?,?)",
        (ftype, fname, label, duration, max_pos)
    )
    con.commit()
    item_id = cur.lastrowid
    con.close()
    return jsonify({"ok": True, "id": item_id, "type": ftype})

@app.route("/api/playlist/<int:pid>", methods=["PATCH"])
def update_playlist_item(pid):
    data = request.json or {}
    con = get_db()
    if "label" in data:
        con.execute("UPDATE playlist SET label=? WHERE id=?", (data["label"], pid))
    if "duration" in data:
        con.execute("UPDATE playlist SET duration=? WHERE id=?", (int(data["duration"]), pid))
    if "show_text" in data:
        con.execute("UPDATE playlist SET show_text=? WHERE id=?", (1 if data["show_text"] else 0, pid))
    if "signs" in data:
        con.execute("UPDATE playlist SET signs=? WHERE id=?", (json.dumps(data["signs"]), pid))
    con.commit()
    con.close()
    return jsonify({"ok": True})

@app.route("/api/playlist/<int:pid>", methods=["DELETE"])
def del_playlist_item(pid):
    con = get_db()
    row = con.execute("SELECT filename, type FROM playlist WHERE id=?", (pid,)).fetchone()
    if row:
        if row["filename"]:
            p = PLAYLIST_DIR / row["filename"]
            if p.exists():
                p.unlink()
        con.execute("DELETE FROM playlist WHERE id=?", (pid,))
        con.commit()
    con.close()
    return jsonify({"ok": True})

@app.route("/api/playlist/control", methods=["POST"])
def playlist_control():
    data = request.json or {}
    action = data.get("action")
    if action == "start":
        state["playlist_on"] = True
    elif action == "stop":
        state["playlist_on"] = False
    return jsonify({"ok": True, "playlist_on": state["playlist_on"]})

@app.route("/api/playlist/reorder", methods=["POST"])
def reorder_playlist():
    ids = (request.json or {}).get("order", [])
    con = get_db()
    for pos, pid in enumerate(ids):
        con.execute("UPDATE playlist SET position=? WHERE id=?", (pos, pid))
    con.commit()
    con.close()
    return jsonify({"ok": True})

@app.route("/api/playlist/apply", methods=["POST"])
def apply_playlist():
    state["playlist_version"] += 1
    return jsonify({"ok": True, "version": state["playlist_version"]})

@app.route("/api/playlist/add-horoscope", methods=["POST"])
def add_horoscope_block():
    """Add a horoscope block (up to 3 signs) to the playlist."""
    data = request.json or {}
    signs = data.get("signs", [])[:3]
    label = data.get("label", "Horóscopos")
    duration = int(data.get("duration", 8))
    con = get_db()
    max_pos = con.execute("SELECT COALESCE(MAX(position),0) FROM playlist").fetchone()[0]
    con.execute(
        "INSERT INTO playlist(type,filename,label,show_text,duration,position,signs) "
        "VALUES('horoscope','',?,1,?,?,?)",
        (label, duration, max_pos + 1, json.dumps(signs))
    )
    con.commit()
    con.close()
    return jsonify({"ok": True})

@app.route("/api/playlist/auto-horoscopes", methods=["POST"])
def auto_horoscope_blocks():
    """Remove existing horoscope blocks and re-create enough to show all 12 signs."""
    all_signs = ["aries","tauro","geminis","cancer","leo","virgo",
                 "libra","escorpio","sagitario","capricornio","acuario","piscis"]
    per_block = 3
    blocks = [all_signs[i:i+per_block] for i in range(0, len(all_signs), per_block)]
    con = get_db()
    con.execute("DELETE FROM playlist WHERE type='horoscope'")
    max_pos = con.execute("SELECT COALESCE(MAX(position),0) FROM playlist").fetchone()[0]
    for i, block_signs in enumerate(blocks):
        names = ", ".join(s.capitalize() for s in block_signs)
        con.execute(
            "INSERT INTO playlist(type,filename,label,show_text,duration,position,signs) "
            "VALUES('horoscope','',?,1,8,?,?)",
            (f"Horóscopos: {names}", max_pos + 1 + i, json.dumps(block_signs))
        )
    con.commit()
    con.close()
    return jsonify({"ok": True, "blocks": len(blocks)})


# ── HOROSCOPES API ─────────────────────────────────────────────────────────────

SIGNS = ["aries","tauro","geminis","cancer","leo","virgo",
         "libra","escorpio","sagitario","capricornio","acuario","piscis"]

@app.route("/api/horoscopes", methods=["GET"])
def get_horoscopes():
    today = time.strftime("%Y-%m-%d")
    con = get_db()
    rows = con.execute(
        "SELECT sign, text, source, date FROM horoscopes WHERE date=? ORDER BY sign", (today,)
    ).fetchall()
    if not rows:
        # fall back to most recent available date
        last = con.execute("SELECT MAX(date) as d FROM horoscopes").fetchone()
        if last and last["d"]:
            rows = con.execute(
                "SELECT sign, text, source, date FROM horoscopes WHERE date=? ORDER BY sign",
                (last["d"],)
            ).fetchall()
    con.close()
    result = {r["sign"]: {"text": r["text"], "source": r["source"], "date": r["date"]} for r in rows}
    return jsonify(result)

@app.route("/api/horoscopes/<sign>", methods=["POST"])
def save_horoscope(sign):
    if sign not in SIGNS:
        return jsonify({"error": "invalid sign"}), 400
    data = request.json or {}
    text = data.get("text", "").strip()
    today = time.strftime("%Y-%m-%d")
    con = get_db()
    con.execute(
        "INSERT INTO horoscopes(sign,text,date,source) VALUES(?,?,?,'manual') "
        "ON CONFLICT(sign,date) DO UPDATE SET text=excluded.text, source='manual'",
        (sign, text, today)
    )
    con.commit()
    con.close()
    return jsonify({"ok": True})

@app.route("/api/horoscopes/fetch", methods=["POST"])
def fetch_horoscopes():
    """Fetch today's horoscopes via horoscope-app-api (no key required)."""
    import urllib.request, urllib.error
    today = time.strftime("%Y-%m-%d")
    fetched = []
    errors = []
    sign_map = {
        "aries":"aries","tauro":"taurus","geminis":"gemini","cancer":"cancer",
        "leo":"leo","virgo":"virgo","libra":"libra","escorpio":"scorpio",
        "sagitario":"sagittarius","capricornio":"capricorn","acuario":"aquarius","piscis":"pisces"
    }
    con = get_db()
    for sign_es, sign_en in sign_map.items():
        text = ""
        # source 1: horoscope-app-api (Heroku)
        try:
            url = f"https://horoscope-app-api.herokuapp.com/api/v1/get-horoscope/daily?sign={sign_en}&day=today"
            req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                d = json.loads(resp.read())
            text = (d.get("data") or {}).get("horoscope_data", "")
        except Exception:
            pass
        # source 2: any.ge horoscope
        if not text:
            try:
                url2 = f"https://any.ge/horoscope/api/?sign={sign_en}&lang=en"
                req2 = urllib.request.Request(url2, headers={"User-Agent":"Mozilla/5.0"})
                with urllib.request.urlopen(req2, timeout=8) as resp2:
                    d2 = json.loads(resp2.read())
                text = d2[0].get("text","") if isinstance(d2, list) and d2 else ""
            except Exception:
                pass
        if text:
            con.execute(
                "INSERT INTO horoscopes(sign,text,date,source) VALUES(?,?,?,'auto') "
                "ON CONFLICT(sign,date) DO UPDATE SET text=excluded.text, source='auto'",
                (sign_es, text, today)
            )
            fetched.append(sign_es)
        else:
            errors.append(sign_es)
    con.commit()
    con.close()
    return jsonify({"ok": True, "fetched": fetched, "errors": errors})


# ── HOROSCOPE SOURCES ──────────────────────────────────────────────────────────

@app.route("/api/horoscope-sources", methods=["GET"])
def list_horoscope_sources():
    con = get_db()
    rows = con.execute("SELECT id, name, url, active, created_at FROM horoscope_sources ORDER BY id").fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/horoscope-sources", methods=["POST"])
def add_horoscope_source():
    data = request.json or {}
    name = data.get("name", "").strip()
    url  = data.get("url", "").strip()
    if not name or not url:
        return jsonify({"error": "name and url required"}), 400
    con = get_db()
    cur = con.execute("INSERT INTO horoscope_sources(name,url) VALUES(?,?)", (name, url))
    con.commit()
    new_id = cur.lastrowid
    con.close()
    return jsonify({"ok": True, "id": new_id})

@app.route("/api/horoscope-sources/<int:src_id>", methods=["DELETE"])
def delete_horoscope_source(src_id):
    con = get_db()
    con.execute("DELETE FROM horoscope_sources WHERE id=?", (src_id,))
    con.commit()
    con.close()
    return jsonify({"ok": True})

@app.route("/api/horoscopes/fetch-sources", methods=["POST"])
def fetch_horoscopes_from_sources():
    """Fetch horoscopes from manually configured source URLs."""
    import urllib.request, urllib.error, re as _re
    today = time.strftime("%Y-%m-%d")
    sign_variants = {
        "aries":["aries"],"tauro":["tauro","taurus"],"geminis":["géminis","geminis","gemini"],
        "cancer":["cáncer","cancer"],"leo":["leo"],"virgo":["virgo"],
        "libra":["libra"],"escorpio":["escorpio","scorpio","escorpión"],
        "sagitario":["sagitario","sagittarius"],"capricornio":["capricornio","capricorn"],
        "acuario":["acuario","aquarius"],"piscis":["piscis","pisces"]
    }
    con = get_db()
    sources = con.execute("SELECT id, name, url FROM horoscope_sources WHERE active=1").fetchall()
    fetched, errors = [], []
    for src in sources:
        try:
            req = urllib.request.Request(src["url"], headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            # strip HTML tags
            text_plain = _re.sub(r'<[^>]+>', ' ', html)
            text_plain = _re.sub(r'&[a-z]+;', ' ', text_plain)
            # find **Sign**: text  OR  Sign: text patterns
            for sign_es, variants in sign_variants.items():
                found = ""
                for v in variants:
                    # markdown bold: **Signo**: texto
                    m = _re.search(
                        r'\*\*\s*' + _re.escape(v) + r'\s*\*\*\s*[:\-]\s*(.{20,500}?)(?:\n|\*\*|$)',
                        text_plain, _re.IGNORECASE
                    )
                    if not m:
                        # plain: Signo: texto (after newline or start of line)
                        m = _re.search(
                            r'(?:^|\n)\s*' + _re.escape(v) + r'\s*[:\-]\s*(.{20,500}?)(?:\n|$)',
                            text_plain, _re.IGNORECASE | _re.MULTILINE
                        )
                    if m:
                        found = m.group(1).strip()
                        break
                if found:
                    con.execute(
                        "INSERT INTO horoscopes(sign,text,date,source) VALUES(?,?,?,?) "
                        "ON CONFLICT(sign,date) DO UPDATE SET text=excluded.text, source=excluded.source",
                        (sign_es, found, today, src["name"])
                    )
                    if sign_es not in fetched:
                        fetched.append(sign_es)
        except Exception as e:
            errors.append({"source": src["name"], "error": str(e)})
    con.commit()
    con.close()
    return jsonify({"ok": True, "fetched": fetched, "errors": errors})


# ── BRAND API ──────────────────────────────────────────────────────────────────

@app.route("/api/brand", methods=["GET"])
def list_brand():
    allowed = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    files = [f.name for f in sorted(BRAND_DIR.glob("*")) if f.suffix.lower() in allowed]
    return jsonify(files)

@app.route("/api/brand", methods=["POST"])
def upload_brand():
    if "image" not in request.files:
        return jsonify({"error": "no image"}), 400
    f = request.files["image"]
    fname = secure_filename(f.filename)
    allowed = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    if Path(fname).suffix.lower() not in allowed:
        return jsonify({"error": "formato no soportado"}), 400
    (BRAND_DIR / fname).parent.mkdir(parents=True, exist_ok=True)
    f.save(str(BRAND_DIR / fname))
    return jsonify({"ok": True, "name": fname})

@app.route("/api/brand/<fname>", methods=["DELETE"])
def del_brand(fname):
    p = BRAND_DIR / secure_filename(fname)
    if p.exists():
        p.unlink()
    return jsonify({"ok": True})


# ── VIDEO UPLOAD ───────────────────────────────────────────────────────────────

@app.route("/api/upload-video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "no video"}), 400
    f = request.files["video"]
    fname = secure_filename(f.filename)
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".ts", ".flv"}
    if Path(fname).suffix.lower() not in allowed:
        return jsonify({"error": "formato no soportado"}), 400
    path = VIDEOS_DIR / fname
    f.save(str(path))
    return jsonify({"ok": True, "path": str(path)})


# ── CUSTOMERS API ──────────────────────────────────────────────────────────────

@app.route("/api/customers", methods=["GET"])
def list_customers():
    con = get_db()
    rows = con.execute("SELECT id, name, face_path, created_at FROM customers ORDER BY name").fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/customers", methods=["POST"])
def add_customer():
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"error": "nombre requerido"}), 400
    if "face" not in request.files:
        return jsonify({"error": "foto requerida"}), 400
    f = request.files["face"]
    fname = secure_filename(f.filename)
    path = FACES_DIR / fname
    f.save(str(path))

    img = cv2.imread(str(path))
    # Crop to selected face bbox if provided
    if img is not None:
        try:
            fx = int(request.form.get("face_x", 0))
            fy = int(request.form.get("face_y", 0))
            fw = int(request.form.get("face_w", 0))
            fh = int(request.form.get("face_h", 0))
            if fw > 0 and fh > 0:
                ih, iw = img.shape[:2]
                img = img[max(0,fy):min(ih,fy+fh), max(0,fx):min(iw,fx+fw)]
        except Exception:
            pass

    emb = compute_embedding(img) if img is not None and img.size > 0 else None

    # Duplicate check
    if emb:
        con = get_db()
        existing = con.execute("SELECT id, name, embedding FROM customers WHERE embedding IS NOT NULL").fetchall()
        con.close()
        for row in existing:
            try:
                stored = json.loads(row["embedding"])
                if cosine_sim(emb, stored) >= 0.35:
                    path.unlink(missing_ok=True)
                    return jsonify({"error": f"Este rostro ya está registrado como \"{row['name']}\"", "duplicate": True}), 409
            except Exception:
                continue

    emb_json = json.dumps(emb) if emb else None
    con = get_db()
    cur = con.execute(
        "INSERT INTO customers (name, face_path, embedding) VALUES (?,?,?)",
        (name, str(path), emb_json)
    )
    con.commit()
    cid = cur.lastrowid
    con.close()
    return jsonify({"ok": True, "id": cid, "embedding_ok": emb is not None})

@app.route("/api/customers/<int:cid>", methods=["DELETE"])
def del_customer(cid):
    con = get_db()
    con.execute("DELETE FROM customers WHERE id=?", (cid,))
    con.commit()
    con.close()
    return jsonify({"ok": True})


# ── DISHES API ─────────────────────────────────────────────────────────────────

@app.route("/api/dishes", methods=["GET"])
def list_dishes():
    con = get_db()
    rows = con.execute("SELECT * FROM dishes ORDER BY name").fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/dishes", methods=["POST"])
def add_dish():
    name = request.form.get("name", "").strip()
    desc = request.form.get("description", "").strip()
    img_path = None
    if "image" in request.files:
        f = request.files["image"]
        if f.filename:
            fname = secure_filename(f.filename)
            p = DISHES_DIR / fname
            f.save(str(p))
            img_path = str(p)
    con = get_db()
    cur = con.execute(
        "INSERT INTO dishes (name, description, image_path) VALUES (?,?,?)",
        (name, desc, img_path)
    )
    con.commit()
    did = cur.lastrowid
    con.close()
    return jsonify({"ok": True, "id": did})

@app.route("/api/dishes/<int:did>", methods=["PUT"])
def edit_dish(did):
    name = request.form.get("name", "").strip()
    desc = request.form.get("description", "").strip()
    con = get_db()
    if "image" in request.files:
        f = request.files["image"]
        if f.filename:
            fname = secure_filename(f.filename)
            p = DISHES_DIR / fname
            f.save(str(p))
            con.execute(
                "UPDATE dishes SET name=?, description=?, image_path=? WHERE id=?",
                (name, desc, str(p), did)
            )
        else:
            con.execute("UPDATE dishes SET name=?, description=? WHERE id=?", (name, desc, did))
    else:
        con.execute("UPDATE dishes SET name=?, description=? WHERE id=?", (name, desc, did))
    con.commit()
    con.close()
    return jsonify({"ok": True})

@app.route("/api/dishes/<int:did>", methods=["DELETE"])
def del_dish(did):
    con = get_db()
    con.execute("DELETE FROM dishes WHERE id=?", (did,))
    con.commit()
    con.close()
    return jsonify({"ok": True})


# ── ORDERS API ─────────────────────────────────────────────────────────────────

@app.route("/api/orders", methods=["POST"])
def add_order():
    data = request.json or {}
    cid = data.get("customer_id")
    did = data.get("dish_id")
    if not cid or not did:
        return jsonify({"error": "customer_id y dish_id requeridos"}), 400
    con = get_db()
    con.execute("INSERT INTO orders (customer_id, dish_id) VALUES (?,?)", (cid, did))
    con.commit()
    con.close()
    return jsonify({"ok": True})

@app.route("/api/orders", methods=["GET"])
def list_orders():
    con = get_db()
    rows = con.execute("""
        SELECT o.id, c.name as customer, d.name as dish, o.ordered_at
        FROM orders o
        JOIN customers c ON c.id = o.customer_id
        JOIN dishes d ON d.id = o.dish_id
        ORDER BY o.ordered_at DESC LIMIT 50
    """).fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])


# ── FACE DETECT ENDPOINT (admin preview) ──────────────────────────────────────

@app.route("/api/detect-faces", methods=["POST"])
def detect_faces():
    if "image" not in request.files:
        return jsonify({"error": "no image"}), 400
    buf = np.frombuffer(request.files["image"].read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"faces": []})
    return jsonify({"faces": detect_faces_in_image(img)})


# ── STATIC FILES ───────────────────────────────────────────────────────────────

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(str(BASE / "uploads"), filename)


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    # Pre-load InsightFace at startup so first recognition is fast
    threading.Thread(target=get_face_app, daemon=True).start()
    print("=" * 52)
    print("  TuPlatilloFue")
    print("  Admin:    http://localhost:5151/admin")
    print("  Pantalla: http://localhost:5151/pantalla")
    print("=" * 52)
    serve(app, host="0.0.0.0", port=5151, threads=8)
