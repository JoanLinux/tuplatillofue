# TuPlatilloFue 🍽️

Sistema de reconocimiento facial para restaurantes. Identifica clientes al llegar, muestra su platillo favorito en pantalla TV y gestiona pedidos — todo sin internet.

---

## ¿Qué hace?

| Pantalla | Función |
|----------|---------|
| `/pantalla` | TV del restaurante — muestra nombre del cliente + platillo favorito + playlist de imágenes/video |
| `/admin` | Panel de administración — cámaras, clientes, platillos, pedidos |

**Flujo:**
1. Cliente entra → cámara lo detecta (YOLOv8 + InsightFace)
2. Sistema lo reconoce por embedding facial
3. Pantalla TV muestra bienvenida personalizada con su platillo favorito
4. Admin registra el pedido

---

## Stack

- **Python 3.10+** con Flask + Waitress
- **YOLOv8n** — detección de personas (Ultralytics)
- **InsightFace** (buffalo_sc) — reconocimiento facial por embeddings
- **SQLite** — clientes, platillos, pedidos, playlist
- **OpenCV** — captura de cámara / video
- **100% offline** — modelos bundleados, sin llamadas a internet

---

## Instalación

```bash
# 1. Clonar
git clone https://github.com/JoanLinux/tuplatillofue.git
cd tuplatillofue

# 2. Entorno virtual
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Dependencias
pip install flask flask-cors waitress ultralytics insightface onnxruntime opencv-python numpy

# 4. Modelos (descargar manualmente y colocar en models/)
# - yolov8n.pt → raíz del proyecto
# - InsightFace buffalo_sc → models/insightface/models/buffalo_sc/
#   • det_500m.onnx
#   • w600k_mbf.onnx
# - DeepFace haarcascades → models/.deepface/weights/
```

---

## Correr

```bat
start.bat
```

O directamente:

```bash
python server.py
```

Accesos:
- **Admin:** http://localhost:5151/admin
- **Pantalla TV:** http://localhost:5151/pantalla

---

## Estructura

```
tuplatillofue/
├── server.py              # Backend Flask + lógica de visión
├── start.bat              # Lanzador Windows
├── static/
│   ├── admin.html         # Panel de administración
│   └── pantalla.html      # Vista TV del restaurante
├── uploads/
│   ├── faces/             # Fotos de clientes (gitignored)
│   ├── dishes/            # Fotos de platillos (gitignored)
│   ├── playlist/          # Imágenes para slideshow
│   └── videos/            # Videos para playlist (gitignored)
├── models/
│   ├── insightface/       # Modelos InsightFace (gitignored)
│   └── .deepface/         # Haarcascades OpenCV
└── tuplatillo.db          # Base de datos SQLite (gitignored)
```

---

## Base de datos

| Tabla | Descripción |
|-------|-------------|
| `customers` | Clientes registrados con embedding facial |
| `dishes` | Platillos con foto y descripción |
| `orders` | Historial de pedidos por cliente |
| `playlist` | Imágenes/videos para la pantalla TV |

---

## API principales

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/video_feed` | Stream MJPEG de la cámara |
| GET | `/status` | Estado del sistema (running, fps, cliente actual) |
| POST | `/start` | Iniciar detección |
| POST | `/stop` | Detener detección |
| GET | `/api/customers` | Listar clientes |
| POST | `/api/customers` | Registrar cliente + foto |
| GET | `/api/dishes` | Listar platillos |
| POST | `/api/dishes` | Agregar platillo |
| GET | `/api/playlist` | Listar playlist |

---

## Requisitos de hardware

- Cámara USB o IP (también acepta archivo de video para pruebas)
- CPU moderna (Intel i5+ o equivalente) — no requiere GPU
- ~500 MB RAM para modelos

---

## Proyecto relacionado

- [YaLlegué](https://github.com/JoanLinux/yallegue) — checador de asistencia con la misma tecnología
