# SONG-AED-EDA


## Autores

|                                                                             | Nombre                                                                   | GitHub                                                     |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------- |
| ![Mariel](https://github.com/MarielUTEC.png?size=50)                        | [Mariel Carolina Tovar Tolentino](https://github.com/MarielUTEC)         | [@MarielUTEC](https://github.com/MarielUTEC) |
| ![Noemi](https://github.com/NoemiHuarino-utec.png?size=50)                  | [Noemi Alejandra Huarino Anchillo](https://github.com/NoemiHuarino-utec) | [@NoemiHuarino-utec](https://github.com/NoemiHuarino-utec) |
| ![Sergio](https://github.com/SergioF21.png?size=50)                         | [Sergio Luis Fierro Neyra] (https://github.com/SergioF21)                | [@SergioF21] (http://github.com/SergioF21)
| ![Dario](https://github.com/DariusKarius.png?size=50)                       | [Dario Ricardo Nuñez Villacorta] (https://github.com/DariusKarius)       | [@DariusKarius] (https://github.com/DariusKarius)          |



## Comandos de ejecución

Advertencia: Solo se puede ejecutar en Linux

### backend
cd backend/song
g++ GraphBuilder.cpp -o GraphBuilder
nvcc kernel.song.cu -o song
g++ -std=c++17 -O2 -Wall Faiss.cpp -o faiss_demo -lfaiss -lpthread
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

### frontend
cd frontend/ 
python3 -m venv venv
source venv/bin/activate
pip3 install streamlit pandas plotly
streamlit run app.py
