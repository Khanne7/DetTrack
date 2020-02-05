Implementación de YoloV3 con PyTorch para el desarrollo de un sistema de detección y seguimiento de objetos.

Entorno de desarrollo utilizado:
- Ubuntu 18.04
- Python 3.6.9
- Anaconda 4.8.0
- Librerías del archivo de dependencias environment.yml

NECESARIO CONTAR CON ARCHIVOS DE PESOS .weights PARA INCLUIR EN LA CARPETA DEL MODELO A USAR.
El archivo de pesos para el dataset COCO se puede descargar de https://pjreddie.com/media/files/yolov3.weights


OPCION A (via terminal):

Archivos necesarios:
- model/
- videos/
- detector_tracker.py
- net.py
- util.py
- pallete

Intrucciones de uso:

Ejecutar en consola el archivo detector_tracker con python junto con estos posibles argumentos de ajuste:

--source: fuente de la que obtener video (Para dispositivos de captura: 0, 1, etc; para archivo de video: nombre del video guardado en la carpeta videos/). Por defecto: 1.

--confidence: Valor entre 0 y 1 que representa la fiabilidad umbral para la cual el detector debe mostrar la clase encontrada. Por defecto: 0.5

--nms_thresh: Umbral para Non Maximum Suppressison. Por defecto: 0.4

--model: nombre de la carpeta de modelo a utilizar dentro de la carpeta model/. Por defecto: COCO.

--reso_det: resolucion con la que el detector debe analizar la imagen. Por defecto: 416.

--reso_track: resolucion con la que el tracker debe analizar la imagen. Por defecto: 320.

--class: filtrado por clases. Separar las distintas clases con un espacio. No tiene valor por defecto.

--tracker: algoritmo de seguimiento a utilizar. Hay ocho posibilidades: csrt, kcf, boosting, mil, tld, medianflow y moose. Recomendados los dos primeros. Por defecto: kcf.

Una vez ejecutado hacer click en los recuadros de deteccion para hacer un seguimiento del objeto seleccionado. Para volver a la fase de deteccion hacer click en cualquier punto de la imagen.


OPCION B (via gui):

Archivos necesarios:
- Todos los de la opción A menos la carpeta videos/
- gui.py
- design_gui.py

Instrucciones de uso:

Ejecutar el archivo gui.py con python. En la interfaz modificar los ajustes descritos en la OPCION A y pulsar RUN para comenzar la fase de deteccion. Hacer click en los recuadros de deteccion para hacer un seguimiento del objeto seleccionado. Para volver a la fase de deteccion hacer click en cualquier punto de la imagen.



