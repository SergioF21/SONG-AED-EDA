# SONG-AED-EDA


## Autores

|                                                                             | Nombre                                                                   | GitHub                                                     |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------- |
| ![Mariel](https://github.com/MarielUTEC.png?size=50)                        | [Mariel Carolina Tovar Tolentino](https://github.com/MarielUTEC)         | [@MarielUTEC](https://github.com/MarielUTEC) |
| ![Noemi](https://github.com/NoemiHuarino-utec.png?size=50)                  | [Noemi Alejandra Huarino Anchillo](https://github.com/NoemiHuarino-utec) | [@NoemiHuarino-utec](https://github.com/NoemiHuarino-utec) |
| ![Sergio](https://github.com/SergioF21.png?size=50)                         | [Sergio Luis Fierro Neyra] (https://github.com/SergioF21)                | [@SergioF21] (http://github.com/SergioF21)
| ![Dario](https://github.com/DariusKarius.png?size=50)                       | [Dario Ricardo Nuñez Villacorta] (https://github.com/DariusKarius)       | [@DariusKarius] (https://github.com/DariusKarius)          |



## Comandos de ejecución (Linux)

Nota: las instrucciones asumen que estás en un sistema Linux con CUDA (nvcc) instalado. Ajusta los comandos para Windows si es necesario.

Recomendado: ejecuta los comandos desde la raíz del proyecto:
cd /home/sergio/Documents/EDA/Proyecto/SONG-AED-EDA

1) Backend — compilar y generar ejecutables
- Requisitos: nvcc (CUDA toolkit), g++.
- Pasos:

```bash
# ir al directorio del backend
cd backend/song

# Compilar GraphBuilder (si tienes GraphBuilder.cpp)
g++ -std=c++17 -O2 -Wall GraphBuilder.cpp -o GraphBuilder

# Compilar kernel CUDA (archivo: kernel_song.cu)
nvcc -std=c++14 kernel_song.cu -o song

# (Opcional) Compilar demo con Faiss si tienes la librería instalada
g++ -std=c++17 -O2 -Wall Faiss.cpp -o faiss_demo -lfaiss -lpthread

# Si necesitas que el linker encuentre librerías instaladas en /usr/local/lib
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

2) Frontend — entorno virtual y ejecutar Streamlit
- Requisitos: python3, virtualenv (opcional pero recomendado).

```bash
# ir al frontend
cd frontend

# crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate

# instalar dependencias (opcional: crear requirements.txt)
pip install streamlit pandas plotly matplotlib

# ejecutar la app
streamlit run app.py
```

Notas rápidas:
- Asegúrate de que EXECUTABLE_PATH en frontend/app.py apunte al ejecutable compilado (ej. ../backend/song/song o ruta absoluta).
- Si hay errores al ejecutar el binary, revisa permisos (chmod +x song) y rutas relativas desde donde ejecutas Streamlit.
- Para Windows, reemplaza `nvcc`/`g++` comandos por las herramientas equivalentes y ajusta rutas.
