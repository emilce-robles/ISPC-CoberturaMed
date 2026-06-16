# CoberturaMed
<img width="570" height="181" alt="image" src="https://github.com/user-attachments/assets/d33cb883-1fd5-4d35-b47c-003a21081a0f" />

CoberturaMed es un prototipo académico desarrollado para aplicar técnicas de Inteligencia Artificial, Aprendizaje Automático y Procesamiento de Imágenes al análisis de facturas médicas.

El objetivo del proyecto es simular el proceso de gestión de un reintegro médico, automatizando tareas como la detección de campos relevantes dentro de una factura, la extracción de información textual, la validación de datos y la clasificación de solicitudes.

## Tecnologías utilizadas

* Python
* Streamlit
* YOLOv8
* EasyOCR
* Random Forest (Scikit-Learn)
* Pandas
* Google Sheets
* Ollama (versión local)

## Funcionalidades

* Carga de facturas médicas en formato imagen.
* Detección automática de campos mediante YOLOv8.
* Extracción de texto mediante OCR.
* Validación de identidad y consistencia de datos.
* Detección de posibles solicitudes duplicadas o inconsistentes.
* Clasificación automática de solicitudes.
* Consulta de estado mediante código de seguimiento.
* Asistente conversacional para acompañar al usuario durante el proceso.

## Flujo general

1. El usuario ingresa sus datos personales.
2. Selecciona el tipo de gestión.
3. Carga una factura médica.
4. El sistema detecta los campos relevantes mediante visión por computadora.
5. Se extrae el contenido textual utilizando OCR.
6. Se ejecutan validaciones y controles de consistencia.
7. La solicitud es clasificada automáticamente.
8. Se genera un código de seguimiento para futuras consultas.

## Modelos utilizados

### YOLOv8

Utilizado para detectar y localizar campos relevantes dentro de las facturas médicas.

### EasyOCR

Utilizado para extraer el contenido textual de los campos detectados.

### Random Forest

Utilizado como modelo complementario para clasificar solicitudes a partir de información estructurada generada durante el procesamiento.

## Ejecución local

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la aplicación:

```bash
python -m streamlit run app.py
```

## Proyecto académico

Este proyecto fue desarrollado por:

* Emilce Robles
* Ian Marini

como parte de las actividades prácticas de las materias relacionadas con Inteligencia Artificial, Aprendizaje Automático y Procesamiento de Imágenes.
