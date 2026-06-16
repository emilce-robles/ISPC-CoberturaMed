import re
import cv2
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from datetime import datetime, timedelta
import gspread
from google.oauth2.service_account import Credentials

from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime


# =========================
# CONFIGURACIÓN GENERAL
# =========================

st.set_page_config(
    page_title="CoberturaMed",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
/* Contenedor principal */
[data-testid="stMainBlockContainer"] {
    max-width: 1150px;
    margin: 0 auto;
    padding-top: 3rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}

/* Oculta un poco el header superior */
[data-testid="stHeader"] {
    background: transparent;
}

/* Título */
h1 {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
}

/* Botones */
.stButton > button {
    border-radius: 12px;
    height: 46px;
    font-weight: 600;
}

/* Inputs */
[data-testid="stTextInput"] input {
    border-radius: 12px;
    height: 44px;
}

/* Mensajes tipo chat */
[data-testid="stChatMessage"] {
    max-width: 100%;
}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = Path("models/best.pt")
RF_MODEL_PATH = Path("models/random_forest_autorizacion.pkl")


# =========================
# ESTADO DE SESIÓN
# =========================

def init_session():
    defaults = {
        "paso": "inicio",
        "gestion": None,
        "nombre_usuario": "",
        "dni_usuario": "",
        "email_usuario": "",
        "datos_raw": None,
        "datos_limpios": None,
        "detecciones": None,
        "confianza_yolo": 0.0,
        "estado_solicitud": None,
        "respuesta_bot": None,
        "codigo_seguimiento": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session()


# =========================
# FUNCIONES AUXILIARES
# =========================

def normalizar_texto(texto):
    if texto is None:
        return ""
    texto = str(texto).upper().strip()
    texto = re.sub(r"\s+", " ", texto)
    return texto


def normalizar_clave(valor):
    if pd.isna(valor):
        return ""
    return re.sub(r"\s+", "", str(valor)).upper().strip()


def normalizar_numero(valor):
    if pd.isna(valor):
        return ""
    return re.sub(r"\D", "", str(valor))

def traducir_estado(estado):
    estados = {
        "APROBADA": "Aprobada",
        "REVISION": "En revisión administrativa",
        "DENEGADA": "Derivada a auditoría",
        "IDENTIDAD_NO_COINCIDE": "Pendiente de validación de identidad",
        "NO_PROCESABLE": "Documentación insuficiente"
    }

    return estados.get(estado, estado)

def limpiar_texto_ocr(texto):
    if not texto:
        return ""

    texto = str(texto)
    texto = texto.replace("|", "")
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def limpiar_monto(texto):
    if not texto:
        return None

    texto = str(texto).strip()
    texto = texto.replace("|", "")
    texto = texto.replace("O", "0")
    texto = texto.replace("o", "0")
    texto = re.sub(r"\s+", "", texto)

    patrones = re.findall(
        r"\d{1,3}(?:,\d{3})+(?:\.\d{2})?|\d+(?:\.\d{2})?",
        texto
    )

    if not patrones:
        return None

    valor = patrones[-1].replace(",", "")

    try:
        return float(valor)
    except Exception:
        return None


def formatear_monto_texto(texto):
    monto = limpiar_monto(texto)

    if monto is None:
        return texto

    return f"{monto:,.2f}"


def detectar_fecha(texto):
    fechas = re.findall(r"\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}", str(texto))
    return fechas[0] if fechas else None


def extraer_nombre_general(texto):
    texto = str(texto).replace("\n", " ")
    texto = re.sub(r"\s+", " ", texto).strip()

    # Caso 1: NOMBRE: CARLOS DAVID GUZMAN VILLADA
    patron = re.search(
        r"NOMBRE[:\s]*([A-ZÁÉÍÓÚÑ ]{5,80})",
        texto,
        re.IGNORECASE
    )

    if patron:
        nombre = patron.group(1).strip()

        # Cortar si el OCR pegó campos siguientes
        nombre = re.split(
            r"\b(TELEFONO|TELÉFONO|DIRECCION|DIRECCIÓN|MUNICIPIO|EDAD|NIT|EMPRESA)\b",
            nombre,
            flags=re.IGNORECASE
        )[0].strip()

        if len(nombre.split()) >= 2:
            return nombre

    # Caso 2: si aparece PACIENTE: CC-1106772747-CARLOS DAVID...
    patron_paciente = re.search(
        r"PACIENTE[:\s-]+(?:CC[-\s]*)?\d{5,15}[-\s]+([A-ZÁÉÍÓÚÑ ]{5,80})",
        texto,
        re.IGNORECASE
    )

    if patron_paciente:
        nombre = patron_paciente.group(1).strip()
        nombre = re.split(
            r"\b(INGRESO|ABONO|PACIENTE|AUTORIZACION|AUTORIZACIÓN)\b",
            nombre,
            flags=re.IGNORECASE
        )[0].strip()

        if len(nombre.split()) >= 2:
            return nombre

    return None


def extraer_cc_paciente(texto):
    texto = str(texto)

    patron = re.search(r"CC\s*[-:]?\s*(\d{5,15})", texto, re.IGNORECASE)

    if patron:
        return patron.group(1)

    return None


def extraer_descripcion_general(texto):
    texto = str(texto).replace("\n", " ")
    texto = re.sub(r"\s+", " ", texto)

    patron_consulta = re.search(
        r"(CONSULTA\s+.+?GENERAL)",
        texto,
        re.IGNORECASE
    )

    if patron_consulta:
        return patron_consulta.group(1).strip()

    patron_linea = re.search(
        r"\b\d{5,8}\b\s+([A-ZÁÉÍÓÚÑ\s]+?)\s+\d{1,3}[,.]\d{3}[,.]\d{2}",
        texto,
        re.IGNORECASE
    )

    if patron_linea:
        return patron_linea.group(1).strip()

    return None


def limpiar_descripcion(texto):
    if not texto:
        return None

    texto = str(texto)
    texto = re.sub(r"\d{1,3}[,.]\d{3}[,.]\d{2}", "", texto)
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def validar_email(email):
    return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email or "") is not None


# =========================
# MODELOS
# =========================

def cargar_yolo():
    if MODEL_PATH.exists():
        return YOLO(str(MODEL_PATH))
    return None


@st.cache_resource(show_spinner=False)
def cargar_ocr():
    import easyocr
    return easyocr.Reader(["es"], gpu=False)


def crear_random_forest_sintetico():
    np.random.seed(42)
    registros = []

    for _ in range(500):
        campos_detectados = np.random.randint(0, 13)
        confianza_yolo = np.random.uniform(0.2, 0.99)
        tiene_nombre = np.random.randint(0, 2)
        tiene_fecha = np.random.randint(0, 2)
        tiene_total = np.random.randint(0, 2)
        tiene_codigo = np.random.randint(0, 2)
        tiene_autorizacion = np.random.randint(0, 2)
        total_coherente = np.random.randint(0, 2)

        if campos_detectados <= 2 or confianza_yolo < 0.35:
            estado = 0
        elif (
            tiene_nombre == 1
            and tiene_fecha == 1
            and tiene_total == 1
            and tiene_codigo == 1
            and total_coherente == 1
            and confianza_yolo >= 0.60
        ):
            estado = 2
        else:
            estado = 1

        registros.append([
            campos_detectados,
            confianza_yolo,
            tiene_nombre,
            tiene_fecha,
            tiene_total,
            tiene_codigo,
            tiene_autorizacion,
            total_coherente,
            estado
        ])

    columnas = [
        "campos_detectados",
        "confianza_yolo",
        "tiene_nombre",
        "tiene_fecha",
        "tiene_total",
        "tiene_codigo",
        "tiene_autorizacion",
        "total_coherente",
        "estado"
    ]

    df = pd.DataFrame(registros, columns=columnas)

    X = df.drop(columns=["estado"])
    y = df["estado"]

    modelo = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=5
    )

    modelo.fit(X, y)

    RF_MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(modelo, RF_MODEL_PATH)

    return modelo


def cargar_random_forest():
    if RF_MODEL_PATH.exists():
        return joblib.load(RF_MODEL_PATH)

    return crear_random_forest_sintetico()


modelo_yolo = cargar_yolo()
lector_ocr = cargar_ocr()
modelo_rf = cargar_random_forest()


# =========================
# PROCESAMIENTO FACTURA
# =========================

def procesar_con_yolo_y_ocr(imagen_pil, modelo_yolo, lector_ocr):
    imagen_np = np.array(imagen_pil.convert("RGB"))

    texto_general_ocr = lector_ocr.readtext(imagen_np, detail=0)
    texto_general_ocr = limpiar_texto_ocr(" ".join(texto_general_ocr).strip())

    datos_extraidos = {"texto_general": texto_general_ocr}
    detecciones = []

    if modelo_yolo is None:
        return datos_extraidos, detecciones, 0.0

    resultados = modelo_yolo.predict(imagen_np, conf=0.25, verbose=False)
    confianzas = []

    for result in resultados:
        names = result.names

        for box in result.boxes:
            cls_id = int(box.cls[0])
            clase = normalizar_texto(names[cls_id]).lower()
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w = imagen_np.shape[:2]
            padding = 8

            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            recorte = imagen_np[y1:y2, x1:x2]

            if recorte.size == 0:
                continue

            texto_ocr = lector_ocr.readtext(recorte, detail=0)
            texto_ocr = limpiar_texto_ocr(" ".join(texto_ocr).strip())

            if texto_ocr and (clase not in datos_extraidos or not datos_extraidos[clase]):
                datos_extraidos[clase] = texto_ocr

            detecciones.append({
                "campo": clase,
                "texto": texto_ocr,
                "confianza": round(conf, 3),
                "bbox": [x1, y1, x2, y2]
            })

            confianzas.append(conf)

    confianza_promedio = float(np.mean(confianzas)) if confianzas else 0.0

    return datos_extraidos, detecciones, confianza_promedio


def completar_datos_desde_texto(datos):
    texto_general = " ".join([str(v) for v in datos.values()])

    resultado = {}

    nombre_factura = extraer_nombre_general(texto_general)
    if not nombre_factura:
        posible_paciente = datos.get("paciente")
        if posible_paciente and not re.search(r"\d", str(posible_paciente)):
            nombre_factura = posible_paciente

    cc_factura = extraer_cc_paciente(texto_general)

    resultado["nombre_factura"] = nombre_factura
    resultado["dni_factura"] = cc_factura or normalizar_numero(datos.get("paciente"))

    resultado["historia"] = datos.get("historia") or datos.get("ingreso")
    resultado["fecha"] = datos.get("fecha") or detectar_fecha(texto_general)
    resultado["autorizacion"] = datos.get("autorizacion")
    resultado["descripcion"] = datos.get("descripcion") or extraer_descripcion_general(texto_general)
    resultado["descripcion"] = limpiar_descripcion(resultado["descripcion"])
    resultado["codigo"] = datos.get("codigo")

    resultado["valor_unitario"] = datos.get("valor_unitario")
    resultado["total"] = datos.get("total")
    resultado["copago"] = datos.get("copago")
    resultado["cuota_moderadora"] = datos.get("cuota_moderadora")

    for campo in ["valor_unitario", "total", "copago", "cuota_moderadora"]:
        if resultado.get(campo):
            resultado[campo] = formatear_monto_texto(resultado[campo])

    resultado["monto_total_num"] = limpiar_monto(resultado["total"])
    resultado["valor_unitario_num"] = limpiar_monto(resultado["valor_unitario"])

    if (
        resultado["monto_total_num"] == 0
        and resultado["valor_unitario_num"] is not None
    ):
        resultado["total"] = resultado["valor_unitario"]
        resultado["monto_total_num"] = resultado["valor_unitario_num"]

    cantidad_detectada = datos.get("cantidad")

    if cantidad_detectada:
        nums = re.findall(r"\b\d+\b", str(cantidad_detectada))
        resultado["cantidad"] = nums[0] if nums else cantidad_detectada
    else:
        resultado["cantidad"] = None

    if not resultado["cantidad"] or str(resultado["cantidad"]).strip() in ["0", "0.00"]:
        if (
            resultado["monto_total_num"] is not None
            and resultado["valor_unitario_num"] is not None
            and abs(resultado["monto_total_num"] - resultado["valor_unitario_num"]) < 1
        ):
            resultado["cantidad"] = "1"

    try:
        cantidad_num = float(re.findall(r"\d+", str(resultado["cantidad"]))[0])
    except Exception:
        cantidad_num = None

    resultado["cantidad_num"] = cantidad_num

    if (
        resultado["monto_total_num"] is not None
        and resultado["valor_unitario_num"] is not None
        and cantidad_num is not None
    ):
        esperado = resultado["valor_unitario_num"] * cantidad_num
        resultado["total_coherente"] = abs(esperado - resultado["monto_total_num"]) < 1
    else:
        resultado["total_coherente"] = False

    return resultado


# =========================
# VALIDACIONES
# =========================

def clasificar_solicitud(datos_limpios, detecciones, confianza_yolo, modelo_rf):
    campos_detectados = len(detecciones)

    features = pd.DataFrame([{
        "campos_detectados": campos_detectados,
        "confianza_yolo": confianza_yolo,
        "tiene_nombre": int(bool(datos_limpios.get("nombre_factura"))),
        "tiene_fecha": int(bool(datos_limpios.get("fecha"))),
        "tiene_total": int(bool(datos_limpios.get("total"))),
        "tiene_codigo": int(bool(datos_limpios.get("codigo"))),
        "tiene_autorizacion": int(bool(datos_limpios.get("autorizacion"))),
        "total_coherente": int(datos_limpios.get("total_coherente", False))
    }])

    pred = modelo_rf.predict(features)[0]

    mapa = {
        0: "NO_PROCESABLE",
        1: "REVISION",
        2: "APROBADA"
    }

    return mapa[pred], features


def detectar_factura_sospechosa(datos_limpios):
    historial = cargar_solicitudes_google_sheets()

    if historial.empty:
        return False

    autorizacion_actual = normalizar_numero(datos_limpios.get("autorizacion"))
    codigo_actual = normalizar_numero(datos_limpios.get("codigo"))
    descripcion_actual = normalizar_clave(datos_limpios.get("descripcion"))
    total_actual = float(datos_limpios.get("monto_total_num") or 0)

    columnas_necesarias = ["autorizacion", "codigo", "descripcion", "monto_total_num"]

    for col in columnas_necesarias:
        if col not in historial.columns:
            return False

    historial["autorizacion_norm"] = historial["autorizacion"].apply(normalizar_numero)
    historial["codigo_norm"] = historial["codigo"].apply(normalizar_numero)
    historial["descripcion_norm"] = historial["descripcion"].apply(normalizar_clave)

    coincidencias = historial[
        (historial["autorizacion_norm"] == autorizacion_actual)
        & (historial["codigo_norm"] == codigo_actual)
        & (historial["descripcion_norm"] == descripcion_actual)
    ]

    if coincidencias.empty:
        return False

    for _, row in coincidencias.iterrows():
        try:
            total_anterior = float(row.get("monto_total_num") or 0)
        except Exception:
            total_anterior = 0

        if abs(total_actual - total_anterior) > 1:
            return True

    return False


def validar_identidad_usuario(datos_limpios):
    nombre_usuario = normalizar_clave(st.session_state.nombre_usuario)
    nombre_factura = normalizar_clave(datos_limpios.get("nombre_factura"))

    dni_usuario = normalizar_numero(st.session_state.dni_usuario)
    dni_factura = normalizar_numero(datos_limpios.get("dni_factura"))

    nombre_ok = nombre_usuario == nombre_factura if nombre_factura else False
    dni_ok = dni_usuario == dni_factura if dni_factura else False

    return nombre_ok, dni_ok


# =========================
# IA GENERATIVA
# =========================


def generar_respuesta_bot(estado, datos_limpios, contexto_extra=""):

    if estado == "APROBADA":
        contexto_extra += """
        La documentación fue recibida correctamente.
        La solicitud quedó registrada.
        """

    elif estado == "REVISION":
        contexto_extra += """
        La solicitud necesita una revisión administrativa adicional.
        Todavía no fue aprobada ni rechazada.
        """

    elif estado == "DENEGADA":
        contexto_extra += """
        Se detectó una inconsistencia en la documentación.
        La solicitud será revisada por auditoría.
        """

    elif estado == "IDENTIDAD_NO_COINCIDE":
        contexto_extra += """
        Los datos ingresados no coinciden con los datos detectados en la factura.
        """

    fecha_actual = datetime.now().strftime("%d/%m/%Y")

    prompt = f"""
Sos CoberturaMed, un asistente virtual de una obra social.

Comportamiento general:
- Actuás como un asesor virtual de CoberturaMed.
- Tu objetivo es ayudar al usuario a comprender el estado de su solicitud.
- Respondé siempre de forma clara, cordial y empática.
- Respondé como si estuvieras conversando con una persona real.
- Utilizá lenguaje natural y cercano.
- Hablale directamente a la persona usando "tu solicitud", "tu factura" y "tus datos".
- No hables del usuario en tercera persona.
- No repitas información que ya fue comunicada previamente.
- No repitas el código de seguimiento salvo que el usuario lo solicite.
- Si el usuario está confundido, explicá la situación con palabras simples.
- Si el usuario expresa preocupación, frustración o dudas, reconocé primero su inquietud antes de responder.
- Si el usuario pregunta qué debe hacer, indicá pasos concretos.
- Si el usuario pregunta si puede hablar con una persona, indicá que puede comunicarse con el área administrativa correspondiente.
- Si el usuario solicita ayuda adicional, brindá orientación práctica dentro de la información disponible.
- Si no tenés información suficiente para responder algo, explicalo de forma transparente.
- Nunca inventes datos.
- Nunca inventes causas que no estén indicadas en la información recibida.
- Evitá respuestas robóticas o excesivamente formales.

Información interna:
El resultado de la solicitud ya fue determinado por el sistema.
No debes modificarlo.
Tu tarea es únicamente explicarle el resultado al usuario de forma clara y humana.
Nunca expliques reglas internas ni procesos del sistema.

No inventes datos.
No prometas pagos.
No digas que el reintegro ya fue realizado.

No menciones palabras técnicas como:
- reglas
- Machine Learning
- Inteligencia Artificial
- Random Forest
- YOLO
- OCR
- validación del modelo
- clasificación automática
- estado interno

Resultado interno del sistema:
{estado}

Este dato es interno y no debe mencionarse literalmente al usuario.

Datos del usuario:
- Nombre declarado: {st.session_state.nombre_usuario}
- DNI declarado: {st.session_state.dni_usuario}
- Email: {st.session_state.email_usuario}

Datos detectados en la factura:
- Paciente: {datos_limpios.get("nombre_factura")}
- DNI/CC de factura: {datos_limpios.get("dni_factura")}
- Fecha de factura: {datos_limpios.get("fecha")}
- Autorización: {datos_limpios.get("autorizacion")}
- Servicio: {datos_limpios.get("descripcion")}
- Código: {datos_limpios.get("codigo")}
- Cantidad: {datos_limpios.get("cantidad")}
- Valor unitario: {datos_limpios.get("valor_unitario")}
- Total: {datos_limpios.get("total")}

Fecha actual real:
{fecha_actual}

Contexto adicional:
{contexto_extra}

Interpretación del resultado:

Si el resultado es APROBADA:
- Informá que la documentación fue recibida correctamente.
- Informá que el caso fue registrado.
- Informá que recibirá novedades dentro de las próximas 24 horas.
- No digas que el reintegro ya fue realizado.

Si el resultado es REVISION:
- Informá que la solicitud requiere una verificación administrativa adicional.
- Explicá que esto no implica necesariamente un rechazo.
- Informá que recibirá una respuesta dentro de los próximos 7 días hábiles.
- No inventes motivos específicos si no fueron informados.

Si el resultado es DENEGADA:
- Informá que se detectó una inconsistencia en la documentación.
- Explicá que el caso será derivado a auditoría.
- Mantené un tono cordial y empático.

Si el resultado es IDENTIDAD_NO_COINCIDE:
- Informá que los datos ingresados no coinciden con los datos detectados en la factura.
- Indicá que puede corregir los datos o cargar otra factura.

Si el resultado es NO_PROCESABLE:
- Solicitá una imagen más clara o una nueva copia de la factura.

Si el usuario realiza una consulta posterior:
- Respondé únicamente la pregunta realizada.
- No vuelvas a explicar todo el caso.
- No repitas el estado completo de la solicitud.
- No repitas información ya mostrada anteriormente.
- Intentá aportar información útil o próximos pasos.
- Si el usuario pregunta por qué ocurrió algo y no existe una causa concreta disponible, explicá que no contás con ese detalle.
- Si el usuario pregunta qué puede hacer ahora, indicá acciones concretas.
- Si el usuario solicita asistencia humana, indicá que puede comunicarse con el área administrativa correspondiente.

Respondé en español.
Respondé en un único párrafo breve.
Respondé solamente la consulta del usuario, sin repetir el resumen completo del caso.
"""

    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )

        texto = response.output_text.strip()

        if texto:
            return texto

    except Exception:
        pass

    respuestas = {
        "APROBADA": "La solicitud fue registrada correctamente. Te avisaremos dentro de las próximas 24 horas para coordinar el reintegro.",
        "REVISION": "La factura requiere una validación adicional. Será revisada por el área administrativa dentro de los próximos 7 días hábiles.",
        "DENEGADA": "La solicitud no puede aprobarse automáticamente porque se detectó una inconsistencia. El caso será derivado a auditoría.",
        "IDENTIDAD_NO_COINCIDE": "Los datos declarados no coinciden con la información detectada en la factura. Podés editar tus datos o cargar otra factura.",
        "NO_PROCESABLE": "No fue posible procesar la factura. Por favor subí una imagen más clara y completa."
    }

    return respuestas.get(estado, "La solicitud fue procesada.")


def mensaje_estado_usuario(estado):
    if estado == "APROBADA":
        return (
            "Tu solicitud fue registrada correctamente. "
            "Te contactaremos dentro de las próximas 24 horas para continuar con la coordinación del reintegro."
        )

    if estado == "REVISION":
        return (
            "Tu solicitud fue registrada correctamente y quedará en revisión administrativa. "
            "Vamos a analizar la documentación durante los próximos 7 días hábiles."
        )

    if estado == "DENEGADA":
        return (
            "No podemos aprobar automáticamente esta solicitud porque detectamos una inconsistencia en la documentación presentada. "
            "El caso será derivado a auditoría para una revisión más detallada."
        )

    if estado == "IDENTIDAD_NO_COINCIDE":
        return (
            "Los datos ingresados no coinciden con los datos detectados en la factura. "
            "Podés corregir tus datos o cargar otra factura."
        )

    return (
        "No pudimos procesar correctamente la factura. "
        "Por favor, subí una imagen más clara y completa para continuar."
    )

# =========================
# GOOGLE SHEETS
# =========================

SHEET_ID = "1oSr7EcG44lr9p9Tsm4LyUWl5bhyk8IxHgJxqzYkCG08"

COLUMNAS_SOLICITUDES = [
    "fecha_registro",
    "codigo_seguimiento",
    "estado_solicitud",
    "nombre_usuario",
    "dni_usuario",
    "email_usuario",
    "nombre_factura",
    "dni_factura",
    "historia",
    "fecha_factura",
    "autorizacion",
    "descripcion",
    "codigo",
    "cantidad",
    "valor_unitario",
    "total",
    "copago",
    "cuota_moderadora",
    "monto_total_num",
    "valor_unitario_num",
    "cantidad_num",
    "total_coherente",
    "campos_detectados",
    "confianza_yolo",
]


@st.cache_resource(show_spinner=False)

def conectar_google_sheets():

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    try:
        # Streamlit Cloud
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=scopes,
        )

    except Exception:
        # Local
        creds = Credentials.from_service_account_file(
            "credentials/service_account.json",
            scopes=scopes,
        )

    client = gspread.authorize(creds)

    return client.open_by_key(SHEET_ID).sheet1


def cargar_solicitudes_google_sheets():
    try:
        sheet = conectar_google_sheets()
        registros = sheet.get_all_records()

        if not registros:
            return pd.DataFrame(columns=COLUMNAS_SOLICITUDES)

        return pd.DataFrame(registros)

    except Exception as e:
        st.error(f"No se pudo leer Google Sheets: {e}")
        return pd.DataFrame(columns=COLUMNAS_SOLICITUDES)


def generar_codigo_seguimiento():
    return "CM-" + datetime.now().strftime("%Y%m%d%H%M%S")


def guardar_solicitud_csv(datos_limpios, estado, confianza_yolo, detecciones):
    codigo_seguimiento = generar_codigo_seguimiento()
    st.session_state.codigo_seguimiento = codigo_seguimiento

    registro = {
        "fecha_registro": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "codigo_seguimiento": codigo_seguimiento,
        "estado_solicitud": estado,
        "nombre_usuario": st.session_state.nombre_usuario,
        "dni_usuario": st.session_state.dni_usuario,
        "email_usuario": st.session_state.email_usuario,
        "nombre_factura": datos_limpios.get("nombre_factura"),
        "dni_factura": datos_limpios.get("dni_factura"),
        "historia": datos_limpios.get("historia"),
        "fecha_factura": datos_limpios.get("fecha"),
        "autorizacion": datos_limpios.get("autorizacion"),
        "descripcion": datos_limpios.get("descripcion"),
        "codigo": datos_limpios.get("codigo"),
        "cantidad": datos_limpios.get("cantidad"),
        "valor_unitario": datos_limpios.get("valor_unitario"),
        "total": datos_limpios.get("total"),
        "copago": datos_limpios.get("copago"),
        "cuota_moderadora": datos_limpios.get("cuota_moderadora"),
        "monto_total_num": datos_limpios.get("monto_total_num"),
        "valor_unitario_num": datos_limpios.get("valor_unitario_num"),
        "cantidad_num": datos_limpios.get("cantidad_num"),
        "total_coherente": datos_limpios.get("total_coherente"),
        "campos_detectados": len(detecciones),
        "confianza_yolo": round(confianza_yolo, 4),
    }

    try:
        sheet = conectar_google_sheets()
        fila = [registro.get(col, "") for col in COLUMNAS_SOLICITUDES]
        sheet.append_row(fila, value_input_option="USER_ENTERED")
    except Exception as e:
        st.error(f"No se pudo guardar en Google Sheets: {e}")

    return registro


def consultar_reintegro(dni, codigo_o_autorizacion):
    historial = cargar_solicitudes_google_sheets()

    if historial.empty:
        return pd.DataFrame()

    dni_norm = normalizar_numero(dni)
    consulta_norm = normalizar_numero(codigo_o_autorizacion)
    consulta_texto = str(codigo_o_autorizacion).strip().upper()

    resultado = historial[
        (historial["dni_usuario"].astype(str).apply(normalizar_numero) == dni_norm)
        & (
            (historial["codigo"].astype(str).apply(normalizar_numero) == consulta_norm)
            | (historial["autorizacion"].astype(str).apply(normalizar_numero) == consulta_norm)
            | (historial["codigo_seguimiento"].astype(str).str.upper() == consulta_texto)
        )
    ]

    return resultado

# =========================
# UI HELPERS
# =========================

def bot_msg(texto):
    with st.chat_message("assistant", avatar="🩺"):
        st.write(texto)


def user_msg(texto):
    with st.chat_message("user", avatar="👤"):
        st.write(texto)


def resumen_usuario():
    st.info(
        f"""
**Datos ingresados**

**Nombre y apellido:** {st.session_state.nombre_usuario}  
**DNI:** {st.session_state.dni_usuario}  
**Email:** {st.session_state.email_usuario}
"""
    )


def resumen_factura(datos):
    datos_mostrar = {
        "Nombre en factura": datos.get("nombre_factura"),
        "DNI / CC en factura": datos.get("dni_factura"),
        "Fecha": datos.get("fecha"),
        "Autorización": datos.get("autorizacion"),
        "Servicio / Descripción": datos.get("descripcion"),
        "Código": datos.get("codigo"),
        "Cantidad": datos.get("cantidad"),
        "Valor unitario": datos.get("valor_unitario"),
        "Total": datos.get("total"),
        "Copago": datos.get("copago"),
        "Cuota moderadora": datos.get("cuota_moderadora")
    }

    st.table(pd.DataFrame(
        [
            {"Campo": k, "Valor detectado": v if v else "No detectado"}
            for k, v in datos_mostrar.items()
        ]
    ))


# =========================
# INTERFAZ PRINCIPAL
# =========================

col1, col2 = st.columns([1, 5])

with col1:
    st.image("assets/cobi.png", width=140)

with col2:
    st.title("🩺 CoberturaMed")
    st.caption("Asistente inteligente para gestión de reintegros médicos")

bot_msg(
    "¡Hola! Soy Coby. Voy a ayudarte a gestionar tu reintegro médico de forma rápida y sencilla."
)

if modelo_yolo is None:
    st.warning("No se encontró models/best.pt. La app usará OCR general sin detección YOLO.")


# =========================
# INICIO: TIPO DE GESTIÓN
# =========================

if st.session_state.paso == "inicio":
    bot_msg("¿Qué gestión querés realizar?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Solicitar reintegro", use_container_width=True):
            st.session_state.gestion = "solicitar"
            st.session_state.paso = "pedir_nombre"
            st.rerun()

    with col2:
        if st.button("Consultar reintegro", use_container_width=True):
            st.session_state.gestion = "consultar"
            st.session_state.paso = "consultar_reintegro"
            st.rerun()


# =========================
# CONSULTAR REINTEGRO
# =========================

elif st.session_state.paso == "consultar_reintegro":
    bot_msg("Para consultar tu reintegro, ingresá tu DNI y el código de prestación, autorización o código de seguimiento.")

    with st.form("form_consulta"):
        dni = st.text_input("DNI")
        codigo = st.text_input("Código / Autorización / Código de seguimiento")
        submit = st.form_submit_button("Consultar")

    if submit:
        resultado = consultar_reintegro(dni, codigo)

        if resultado.empty:
            st.warning("No encontré solicitudes asociadas a esos datos.")
        else:
            fila = resultado.iloc[-1]

            st.success("Encontramos la siguiente solicitud:")

            st.markdown(f"""
    **Nombre:** {fila.get("nombre_usuario", "No disponible")}

    **Fecha de factura:** {fila.get("fecha_factura", "No disponible")}

    **Servicio:** {fila.get("descripcion", "No disponible")}

    **Código de autorización:** {fila.get("autorizacion", "No disponible")}

    **Estado actual:** {traducir_estado(fila.get("estado_solicitud", ""))}

    **Código de seguimiento:** {fila.get("codigo_seguimiento", "No disponible")}
    """)

            estado = fila.get("estado_solicitud", "")

            if estado == "APROBADA":
                bot_msg(
                    "Tu solicitud fue aprobada y quedó registrada correctamente. "
                    "Te contactaremos dentro de las próximas 24 horas para coordinar el reintegro."
                )

            elif estado == "REVISION":
                bot_msg(
                    "Tu solicitud está siendo revisada por nuestro equipo administrativo. "
                    "Recibirás una actualización dentro de los próximos 7 días hábiles."
                )

            elif estado == "DENEGADA":
                bot_msg(
                    "Tu solicitud fue derivada a auditoría porque se detectaron inconsistencias en la documentación presentada."
                )

            else:
                bot_msg(
                    "Tu solicitud se encuentra registrada. Si necesitás más información, comunicate con el área administrativa indicando tu código de seguimiento."
                )

    if st.button("Volver al inicio"):
        st.session_state.paso = "inicio"
        st.rerun()


# =========================
# SOLICITAR REINTEGRO - DATOS
# =========================

elif st.session_state.paso == "pedir_nombre":
    bot_msg("Para comenzar, por favor indicame tu nombre y apellido completos.")

    nombre = st.text_input("Nombre y apellido completos")

    if st.button("Continuar"):
        if not nombre.strip():
            st.error("Ingresá tu nombre y apellido.")
        else:
            st.session_state.nombre_usuario = nombre.strip()
            st.session_state.paso = "pedir_dni"
            st.rerun()


elif st.session_state.paso == "pedir_dni":
    user_msg(st.session_state.nombre_usuario)
    bot_msg("Gracias. Ahora ingresá tu DNI.")

    dni = st.text_input("DNI")

    if st.button("Continuar"):
        if not normalizar_numero(dni):
            st.error("Ingresá un DNI válido.")
        else:
            st.session_state.dni_usuario = normalizar_numero(dni)
            st.session_state.paso = "pedir_email"
            st.rerun()


elif st.session_state.paso == "pedir_email":
    user_msg(st.session_state.dni_usuario)
    bot_msg("Perfecto. Ahora indicame tu correo electrónico.")

    email = st.text_input("Correo electrónico")

    if st.button("Continuar"):
        if not validar_email(email):
            st.error("Ingresá un correo electrónico válido.")
        else:
            st.session_state.email_usuario = email.strip()
            st.session_state.paso = "confirmar_datos"
            st.rerun()


elif st.session_state.paso == "confirmar_datos":
    bot_msg("Estos son los datos que registré. ¿Son correctos o querés modificar alguno?")
    resumen_usuario()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Sí, son correctos"):
            st.session_state.paso = "cargar_factura"
            st.rerun()

    with col2:
        if st.button("Modificar nombre"):
            st.session_state.paso = "pedir_nombre"
            st.rerun()

    with col3:
        if st.button("Modificar DNI"):
            st.session_state.paso = "pedir_dni"
            st.rerun()

    with col4:
        if st.button("Modificar email"):
            st.session_state.paso = "pedir_email"
            st.rerun()


# =========================
# CARGA Y PROCESAMIENTO DE FACTURA
# =========================

elif st.session_state.paso == "cargar_factura":
    resumen_usuario()
    bot_msg("Ahora adjuntá la factura médica que querés presentar para reintegro.")

    archivo = st.file_uploader(
        "Subir factura médica",
        type=["jpg", "jpeg", "png"]
    )

    if archivo is not None:
        imagen = Image.open(archivo)

        st.image(imagen, caption="Factura cargada", use_container_width=True)

        with st.spinner("Estamos revisando tu factura, por favor aguardá unos minutos..."):
            datos_raw, detecciones, confianza_yolo = procesar_con_yolo_y_ocr(
                imagen,
                modelo_yolo,
                lector_ocr
            )

            datos_limpios = completar_datos_desde_texto(datos_raw)

        st.session_state.datos_raw = datos_raw
        st.session_state.detecciones = detecciones
        st.session_state.confianza_yolo = confianza_yolo
        st.session_state.datos_limpios = datos_limpios
        st.session_state.paso = "confirmar_factura"
        st.rerun()


elif st.session_state.paso == "confirmar_factura":
    datos_limpios = st.session_state.datos_limpios

    bot_msg("He revisado tu factura y detecté la siguiente información:")
    resumen_factura(datos_limpios)

    bot_msg("¿Es correcta esta información?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Sí, es correcta"):
            with st.spinner("Estamos validando tu solicitud, por favor aguardá unos minutos..."):

                nombre_ok, dni_ok = validar_identidad_usuario(datos_limpios)

                if not nombre_ok or not dni_ok:
                    st.session_state.estado_solicitud = "IDENTIDAD_NO_COINCIDE"
                    st.session_state.respuesta_bot = mensaje_estado_usuario("IDENTIDAD_NO_COINCIDE")
                    st.session_state.paso = "identidad_no_coincide"
                    st.rerun()

                if detectar_factura_sospechosa(datos_limpios):
                    estado = "DENEGADA"
                    features = pd.DataFrame([{
                        "motivo": "Factura previamente registrada con distinto importe"
                    }])
                else:
                    estado, features = clasificar_solicitud(
                        datos_limpios,
                        st.session_state.detecciones,
                        st.session_state.confianza_yolo,
                        modelo_rf
                    )

                st.session_state.estado_solicitud = estado
                st.session_state.features = features
                st.session_state.respuesta_bot = mensaje_estado_usuario(estado)

                guardar_solicitud_csv(
                    datos_limpios,
                    estado,
                    st.session_state.confianza_yolo,
                    st.session_state.detecciones
                )

                st.session_state.paso = "resultado"
                st.rerun()

    with col2:
        if st.button("No, quiero subir otra factura"):
            st.session_state.paso = "cargar_factura"
            st.rerun()


elif st.session_state.paso == "identidad_no_coincide":
    st.error("Los datos declarados no coinciden con la factura.")
    def mensaje_estado_usuario(estado):
        if estado == "APROBADA":
            return (
                "Tu solicitud fue registrada correctamente. "
                "Te contactaremos dentro de las próximas 24 horas para continuar con la coordinación del reintegro."
            )

        if estado == "REVISION":
            return (
                "Tu solicitud fue registrada correctamente, pero necesita una verificación administrativa adicional antes de avanzar. "
                "Esto puede ocurrir cuando el sistema necesita revisar antecedentes, registros previos o detalles administrativos asociados a la factura. "
                "Recibirás una respuesta dentro de los próximos 7 días hábiles."
            )

        if estado == "DENEGADA":
            return (
                "No podemos aprobar automáticamente esta solicitud porque detectamos una inconsistencia en la documentación presentada. "
                "El caso será derivado a auditoría para una revisión más detallada."
            )

        return (
            "No pudimos procesar correctamente la factura. "
            "Por favor, subí una imagen más clara y completa para continuar."
        )

    datos = st.session_state.datos_limpios

    st.write("**Datos declarados:**")
    st.write(f"Nombre: {st.session_state.nombre_usuario}")
    st.write(f"DNI: {st.session_state.dni_usuario}")

    st.write("**Datos detectados en factura:**")
    st.write(f"Nombre: {datos.get('nombre_factura')}")
    st.write(f"DNI/CC: {datos.get('dni_factura')}")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Editar nombre"):
            st.session_state.paso = "pedir_nombre"
            st.rerun()

    with col2:
        if st.button("Editar DNI"):
            st.session_state.paso = "pedir_dni"
            st.rerun()

    with col3:
        if st.button("Subir otra factura"):
            st.session_state.paso = "cargar_factura"
            st.rerun()


# =========================
# RESULTADO FINAL
# =========================

elif st.session_state.paso == "resultado":
    estado = st.session_state.estado_solicitud

    if estado == "APROBADA":
        st.success("✅ Solicitud aprobada")
    elif estado == "REVISION":
        st.warning("🟡 Solicitud requiere revisión")
    elif estado == "DENEGADA":
        st.error("⛔ Solicitud denegada por inconsistencia")
    else:
        st.error("🔴 Factura no procesable")

    st.write(st.session_state.respuesta_bot)

    if st.session_state.codigo_seguimiento:
        st.info(f"Tu código de seguimiento es: **{st.session_state.codigo_seguimiento}**")

    pregunta = st.text_input("¿Tenés alguna consulta sobre el resultado?")

    if st.button("Enviar consulta"):

        if pregunta.strip():

            with st.spinner("Coby está procesando tu consulta..."):

                pregunta_limpia = pregunta.lower().strip()

                # Preguntas simples controladas por la aplicación
                if (
                    "fecha" in pregunta_limpia
                    or "que dia es hoy" in pregunta_limpia
                    or "qué día es hoy" in pregunta_limpia
                    or "hoy es" in pregunta_limpia
                ):

                    respuesta = f"Hoy es {datetime.now().strftime('%d/%m/%Y')}."

                elif "hora" in pregunta_limpia:

                    respuesta = f"La hora actual es {datetime.now().strftime('%H:%M')}."

                elif (
                    "hablar con alguien" in pregunta_limpia
                    or "hablar con una persona" in pregunta_limpia
                    or "atencion al cliente" in pregunta_limpia
                    or "atención al cliente" in pregunta_limpia
                    or "soporte" in pregunta_limpia
                    or "contactar" in pregunta_limpia
                    or "contacto" in pregunta_limpia
                    or "reclamo" in pregunta_limpia
                ):
                    respuesta = (
                        "Entiendo. Si querés hablar con una persona, podés comunicarte con Atención al Cliente "
                        "escribiendo a atencionalcliente@coberturamed.com. "
                        "Te recomiendo incluir tu código de seguimiento para que puedan ubicar tu solicitud más rápido."
                    )

                else:

                    respuesta = generar_respuesta_bot(
                        estado,
                        st.session_state.datos_limpios,

                        contexto_extra=f"""
    La persona preguntó: {pregunta}

    Respondé únicamente a esa pregunta.

    Respondé con tono humano, cordial y empático.

    No saludes nuevamente.
    No repitas el resumen del caso.
    No repitas el código de seguimiento salvo que el usuario lo solicite.
    No menciones reglas internas.
    No menciones estados internos.
    No menciones modelos, OCR, YOLO ni Random Forest.
    No inventes fechas ni días de la semana.
    No inventes información que no figure en los datos recibidos.

    Opciones que puede realizar:
    - No necesita acercarse a ningún lugar por ahora.
    - Puede esperar la respuesta dentro del plazo informado.
    - Puede consultar el estado desde esta misma aplicación usando su DNI y código de seguimiento.
    - Si cree que cargó mal algún dato o subió una factura incorrecta, puede iniciar una nueva gestión.
    - Si necesita hablar con una persona, puede comunicarse con el área administrativa de CoberturaMed e indicar su código de seguimiento: {st.session_state.codigo_seguimiento}.

    Hablale directamente usando expresiones como:
    - "podés"
    - "tu solicitud"
    - "tu factura"
    - "tu caso"

    Si no conocés una respuesta, explicalo de forma transparente.
    """
                )

            bot_msg(respuesta)


    if st.button("Realizar otra gestión"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


st.markdown("---")

st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
        © 2026 CoberturaMed - Prototipo académico desarrollado por
        Emilce Robles e Ian Marini.
        <br>
        Proyecto de Inteligencia Artificial y Aprendizaje Automático.
    </div>
    """,
    unsafe_allow_html=True
)