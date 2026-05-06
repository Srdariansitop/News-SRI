# 📰 NewsIR - News Information Retrieval System

Motor de búsqueda e indexación de noticias con capacidades de recuperación semántica, búsqueda híbrida y generación de respuestas con IA.

---

## 🚀 Inicio Rápido

### Requisitos Previos
- **Python 3.8+**
- **Git**
- **Conexión a internet** (para descargar modelos y acceder a APIs)

### 1️⃣ Clonar el Repositorio

```bash
git clone https://github.com/tuusuario/NewsIR.git
cd NewsIR
```

### 2️⃣ Crear el Archivo `.env`

En la raíz del proyecto, crea un archivo `.env` con la siguiente configuración:

```env
# API Keys
GROQ_API_KEY=your_groq_api_key_here
```

**¿Dónde obtener tu API Key de Groq?**
1. Ve a [https://console.groq.com](https://console.groq.com)
2. Crea una cuenta (es gratis)
3. Genera una API key
4. Cópiala en el archivo `.env`

### 3️⃣ Instalar Dependencias

```bash
# Crear entorno virtual (opcional pero recomendado)
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 4️⃣ Ejecutar la Aplicación

```bash
python app/main.py
```

Se abrirá un menú interactivo con las opciones disponibles.

---

## 📋 Opciones del Menú

### 🔍 Búsqueda y Recuperación
- **1** - Descargar artículos de BBC (Crawling)
- **2** - Descargar + Indexar
- **3** - Generar Embeddings (búsqueda semántica)
- **4** - Buscar con BM25
- **5** - Búsqueda Semántica
- **6** - Búsqueda Híbrida (BM25 + Semántica) 🔥
- **7** - RAG con IA 🤖
- **8** - RAG Completo (Búsqueda Híbrida + Web + IA) ⚡
- **9** - Crawling + Indexing + Embeddings (todo en uno)

### 🌐 Gestión de Documentos Web
- **10** - Ver estadísticas de almacenamiento web
- **11** - Reindexar con documentos web nuevos

### 🧹 Mantenimiento
- **12** - Limpiar duplicados
- **13** - Borrar TODA la base de datos
- **14** - Borrar SOLO documentos
- **15** - Borrar SOLO embeddings

- **16** - Salir del programa

---

## 📁 Estructura de Carpetas

```
NewsIR/
├── app/
│   ├── main.py              ← PUNTO DE ENTRADA
│   ├── crawler/             # Descarga de artículos
│   ├── indexing/            # Construcción de índices
│   ├── vector/              # Embeddings y búsqueda semántica
│   ├── retrieval/           # BM25 y búsqueda híbrida
│   ├── RAG/                 # Sistema de IA con contexto
│   ├── web/                 # Búsqueda en web
│   ├── maintenance/         # Limpieza y gestión de datos
│   └── utils/               # Utilidades
├── data/                    # Carpeta local (NO tracked en git)
│   ├── raw/                 # Documentos descargados
│   ├── index/               # Índices invertidos
│   └── vector_db/           # Base de datos vectorial
├── tests/                   # Tests unitarios
├── requirements.txt         # Dependencias
├── .env                     # Variables de entorno (crea este archivo)
├── .gitignore               # Configuración de git
└── README.md                # Este archivo
```

**Nota:** La carpeta `modelos/` se crea automáticamente si decides almacenar modelos localmente. Si no existe, los modelos se descargan una sola vez y se cachean en `~/.cache/huggingface/`.

---

## ⚙️ Configuración Inicial Recomendada

**Primera vez ejecutando la aplicación:**

1. Ejecuta la opción **9** para descargar, indexar y generar embeddings en un solo paso
   - Esto tomará **varios minutos** la primera vez
   - Se descargará automáticamente el modelo de embeddings (~100MB)

2. Después, ya puedes usar cualquiera de las opciones de búsqueda

---

## 🔄 Flujo Típico de Uso

```
1. Ejecutar: python app/main.py

2. Primera ejecución (opción 9):
   - Descarga artículos de BBC
   - Crea índices de búsqueda
   - Genera embeddings

3. Hacer búsquedas:
   - Opción 6 (Híbrida) - La mejor relación calidad/velocidad
   - Opción 8 (RAG) - Con respuestas sintetizadas por IA

4. (Opcional) Actualizar datos:
   - Opción 1 para más artículos
   - Opción 11 para reindexar con documentos web nuevos
```

---

## ⚠️ Notas Importantes

### Primera Ejecución
- La descarga del modelo de embeddings es **automática** (~100MB, una sola vez)
  - Se descarga en `~/.cache/huggingface/` y no necesita crear carpeta `modelos/`
- El crawling de artículos toma **2-5 minutos** dependiendo de conexión
- La indexación y generación de embeddings toman **3-10 minutos**

### Carpeta `data/`
- Se crea **automáticamente** al ejecutar
- **No está incluida** en el repositorio (git la ignora)
- Contiene: documentos, índices y embeddings

### Modelos de Embeddings
- Se descargan automáticamente la primera vez
- Se cachean en `~/.cache/huggingface/` (no ocupan espacio en tu proyecto)
- **No necesitas hacer nada especial** - todo es automático

### API de Groq
- **Necesaria solo para** opciones 7 y 8 (RAG con IA)
- Es **gratis** con límite de requests diarios
- Si no la configuras, las búsquedas normales (opciones 4-6) funcionan perfectamente

### Requisitos de Red
- Se necesita conexión a internet para:
  - Descargar artículos de BBC (crawling)
  - Búsqueda web (opción 8)
  - Acceder a Groq API (opciones 7-8)

---

