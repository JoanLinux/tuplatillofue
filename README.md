# TuPlatilloFue

Sistema de reconocimiento facial para restaurantes. Detecta clientes al pasar y muestra en pantalla su último platillo ordenado.

## Flujo

```
Cámara → YOLOv8 detecta persona → DeepFace identifica cara
       → SQLite busca último platillo → Pantalla TV muestra imagen + nombre
```

## Accesos

| URL | Qué es |
|-----|--------|
| `http://localhost:5151/admin` | Panel de administración |
| `http://localhost:5151/pantalla` | Pantalla TV (modo kiosco) |

## Inicio rápido

```bat
start.bat
```

O manualmente:

```bash
..\.venv\Scripts\pip install deepface tf-keras
..\.venv\Scripts\python.exe server.py
```

## Dependencias principales

- `ultralytics` — detección de personas (YOLOv8)
- `deepface` + `tf-keras` — reconocimiento facial (modelo Facenet)
- `flask` + `waitress` — servidor web
- `opencv-python` — captura de cámara
- `sqlite3` — base de datos integrada (no requiere instalación)

## Uso

### 1. Registrar platillos (tab Platillos)
- Nombre, descripción y foto del platillo

### 2. Registrar clientes (tab Clientes)
- Nombre + foto frontal clara de la cara
- El sistema extrae el embedding facial automáticamente

### 3. Registrar pedidos (tab Pedidos)
- Selecciona cliente + platillo → Guardar
- El "último pedido" es lo que se mostrará en pantalla

### 4. Iniciar detección
- Ingresa fuente de cámara (`0` = webcam, `1` = segunda cámara, `rtsp://...` = IP cam)
- Clic en **Iniciar**
- La pantalla TV (`/pantalla`) se actualiza automáticamente al detectar un cliente

## Ajustes de precisión

En `server.py`:

```python
DETECT_INTERVAL = 15      # reconocimiento facial cada N frames (bajar = más frecuente)
threshold = 0.75          # similitud mínima para reconocer (subir = más estricto)
```

## Estructura de archivos

```
tuplatillofue/
├── server.py              # Backend Flask
├── start.bat              # Lanzador Windows
├── tuplatillo.db          # SQLite (se crea al iniciar)
├── static/
│   ├── admin.html         # Panel de administración
│   └── pantalla.html      # Pantalla TV fullscreen
└── uploads/
    ├── faces/             # Fotos de clientes
    └── dishes/            # Fotos de platillos
```

## Hardware recomendado

- GPU NVIDIA (CUDA) para inferencia rápida
- Cámara a ~1-2m de altura, apuntando a la cara del cliente
- TV/monitor secundario para la pantalla de atracción
