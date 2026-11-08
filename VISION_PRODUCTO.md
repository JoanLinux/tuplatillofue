# TuPlatilloFue — Visión de Producto y Estrategia de Venta

## Qué es

Sistema de reconocimiento facial en tiempo real que identifica personas al llegar a un punto de atención y muestra automáticamente información personalizada en pantalla — sin que el cliente haga nada.

**Stack técnico:**
- YOLOv8 (detección de personas) — GPU local, sin cloud
- DeepFace / Facenet (reconocimiento facial) — embeddings locales
- SQLite (base de datos de clientes e historial)
- Flask + Python (servidor web)
- HTML/JS (pantallas de display y administración)
- 100% offline — sin suscripción a terceros

---

## MVP actual: Restaurantes

**Nombre comercial sugerido:** TuPlatilloFue  
**Propuesta de valor:** Cliente entra → sistema lo reconoce → pantalla muestra su último platillo favorito → aumenta recompra y experiencia personalizada.

**Flujo:**
1. Admin registra cliente con foto frontal
2. Admin registra platillos con foto y descripción
3. Admin vincula pedidos a clientes
4. Sistema detecta cara → muestra en TV: "¡Bienvenido Juan! Tu favorito: Pasta Alfredo 🍝"

---

## Verticales de mercado

### 🏥 Salud / Hospitales

| Caso de uso | Valor |
|---|---|
| Recepción / admisiones | Paciente llega → historial médico, alergias, médico asignado aparecen antes de que llegue al mostrador |
| Urgencias | Identifica paciente inconsciente → expediente inmediato sin que hable |
| Farmacia hospitalaria | Reconoce paciente → muestra medicamentos activos, contraindicaciones |
| Acceso a áreas restringidas | Solo personal autorizado entra a quirófanos, laboratorios, farmacia |
| Clínicas y consultorios | Paciente frecuente → doctor ya tiene su expediente abierto al llegar |

**Diferenciador vs competencia:** Paige AI (adquirida por Tempus) se enfoca en patología oncológica con IA. Nosotros hacemos reconocimiento de pacientes en punto de contacto — nicho sin cobertura masiva.

---

### 🏨 Hospitalidad / Hoteles

| Caso de uso | Valor |
|---|---|
| Lobby / check-in | Huésped detectado → recepción ve nombre, historial, habitación preferida, peticiones especiales |
| Restaurante del hotel | Sistema integrado platillos + perfil de huésped |
| Acceso a amenidades | Gym, spa, alberca — sin tarjeta llave |
| Concierge proactivo | Staff recibe alerta en tablet cuando llega huésped VIP |

---

### 🏦 Banca y Servicios Financieros

| Caso de uso | Valor |
|---|---|
| Sucursales bancarias | Cliente frecuente detectado → ejecutivo preparado con su portafolio antes de llegar al escritorio |
| Cajas / ventanillas | Fila inteligente — sistema ya sabe qué trámite probablemente trae |
| Acceso a bóvedas / áreas seguras | Control de acceso biométrico sin tarjeta |

---

### 🏫 Educación

| Caso de uso | Valor |
|---|---|
| Control de asistencia | Alumnos pasan frente a cámara → asistencia registrada automáticamente |
| Acceso a instalaciones | Laboratorios, biblioteca — sin credencial física |
| Identificación en exámenes | Verificación de identidad en centros de evaluación |

---

### 🏢 Corporativo / Edificios

| Caso de uso | Valor |
|---|---|
| Control de acceso | Entrada sin tarjeta ni huella — solo cara |
| Registro de visitas | Visitante detectado → notificación automática al anfitrión |
| Tiempo y asistencia | Registro de entrada/salida de empleados |

---

### 🏟️ Eventos / Entretenimiento

| Caso de uso | Valor |
|---|---|
| Estadios / arenas | Lista negra automática, acceso VIP sin boleto físico |
| Conciertos / festivales | Acceso express para abonados |
| Casinos | Identificación de clientes frecuentes, lista de exclusión voluntaria |

---

### 🏛️ Gobierno / Sector Público

| Caso de uso | Valor |
|---|---|
| Ventanillas de trámites | Ciudadano llega → sistema ya tiene expediente listo → reduce tiempo de atención |
| Juzgados / ministerio público | Identificación automática de personas en audiencias |
| Instituciones penitenciarias | Control de acceso e identificación de internos/visitantes |

---

## Competidores por vertical

### Restaurantes
| Competidor | Precio | Face Recognition |
|---|---|---|
| PopMenu | ~$200 USD/mes | ❌ No |
| Toast POS | ~$110 USD/mes | ❌ No |
| SevenRooms | ~$300 USD/mes | ❌ Solo por reserva |
| **TuPlatilloFue** | **$0 recurrente** | **✅ Sí, local** |

### Hospitales
| Competidor | Enfoque | Face Recognition |
|---|---|---|
| Epic Systems | HIS completo, caro | ❌ No en recepción |
| Meditech | HIS | ❌ No |
| AWS Rekognition | API cloud | ✅ Pero cloud, costo por uso |
| **Nuestra solución** | **Reconocimiento en recepción** | **✅ Local, sin cloud** |

### Control de acceso
| Competidor | Precio | |
|---|---|---|
| HikVision FaceID | Hardware dedicado $500-2000 USD | Hardware propietario |
| ZKTeco | Hardware $300-1500 USD | Hardware propietario |
| **Nuestra solución** | **Cámara IP estándar + PC** | Flexible, integrable |

---

## Modelo de negocio sugerido

### Opción A — Licencia por sitio
- Instalación única: $15,000 – $40,000 MXN según vertical
- Soporte anual: $3,000 – $8,000 MXN/año
- Sin costo por usuario ni por reconocimiento

### Opción B — SaaS local (edge)
- Hardware entregado (mini PC + cámara): $8,000 MXN equipo
- Suscripción mensual: $1,500 – $4,000 MXN/mes según vertical
- Incluye actualizaciones y soporte

### Opción C — Por proyecto enterprise
- Hospitales, gobierno, corporativos: cotización a medida
- Integraciones con HIS / POS / ERP: $50,000 – $200,000 MXN proyecto

---

## Roadmap técnico

### V1 — MVP actual ✅
- Detección de personas (YOLOv8)
- Reconocimiento facial (DeepFace/Facenet)
- Base de datos local (SQLite)
- Panel admin web
- Pantalla TV de display

### V2 — Próximas features
- [ ] Integración con POS (Toast, Clip, Bind)
- [ ] App móvil para staff (alertas en tiempo real)
- [ ] Múltiples cámaras simultáneas
- [ ] Modo de registro de asistencia (export Excel/PDF)
- [ ] Dashboard de analytics (horas pico, clientes frecuentes)
- [ ] API REST para integrar con sistemas externos

### V3 — Enterprise
- [ ] Active Directory / LDAP para corporativos
- [ ] Integración HL7/FHIR para hospitales
- [ ] Apertura de puertas / torniquetes (relay GPIO)
- [ ] Multi-tenant (varias sucursales)
- [ ] Encriptación de embeddings faciales (cumplimiento LFPDPPP)

---

## Consideraciones legales (México)

- **LFPDPPP** — Ley Federal de Protección de Datos Personales en Posesión de Particulares
- Los datos biométricos (embeddings faciales) son datos sensibles
- Requiere: aviso de privacidad, consentimiento explícito del titular
- **Ventaja de arquitectura local:** los datos nunca salen del establecimiento — más fácil cumplir que soluciones cloud

---

## Propuesta de pitch para restaurantes

> "¿Cuánto vale que tu cliente sienta que lo recuerdas? TuPlatilloFue identifica a tus clientes frecuentes automáticamente y muestra en pantalla su platillo favorito — sin app, sin loyalty card, sin que hagan nada. Solo entran y la experiencia ya los espera."

## Propuesta de pitch para hospitales

> "Cada segundo en urgencias importa. Nuestro sistema identifica al paciente con una cámara y despliega su expediente clínico antes de que llegue al mostrador — sin que el paciente hable ni lleve documentos."

---

*Desarrollado con Python + YOLOv8 + DeepFace. 100% local. Sin costo por uso.*
