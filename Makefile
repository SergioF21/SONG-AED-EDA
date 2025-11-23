# Compilar
make clean
make

# O manualmente
nvcc -std=c++11 -O3 -arch=sm_70 -use_fast_math main.cu -o song_gpu

# Ejecutar
./song_gpu
```

### 🔑 Características Clave de Stage 2 GPU:

| Característica | Descripción | Beneficio |
|----------------|-------------|-----------|
| **Shared Memory** | Query point en memoria rápida | 20x menos latencia |
| **Warp Reduction** | `__shfl_down_sync()` | Reducción O(log n) |
| **Paralelización** | 256 threads × N candidatos | Procesa miles simultáneamente |
| **Coalescencia** | Accesos consecutivos | Maximiza 900 GB/s bandwidth |
| **Device-to-Device** | Sin transferencias CPU↔GPU | Elimina bottleneck PCIe |

### 💡 Optimizaciones Implementadas:

1. ✅ **Grid-Stride Loop**: Maneja más datos que threads
2. ✅ **Two-Level Reduction**: Warp + Block reduction
3. ✅ **Dynamic Shared Memory**: Ajustable según dimensionalidad
4. ✅ **Atomic-Free**: Sin operaciones atómicas (máximo paralelismo)
5. ✅ **Memory Coalescing**: Accesos optimizados a global memory

### 📊 Output Esperado:
```
=== EJECUTANDO STAGE 2 EN GPU ===
[GPU Stage 2] Configuración del kernel:
        -> Bloques: 4
        -> Threads por bloque: 256
        -> Shared memory: 0.5 KB
        -> Dimensiones: 16
[GPU Stage 2] Kernel ejecutado exitosamente

=== RESUMEN GPU STAGE 2: DISTANCIAS CALCULADAS ===
Candidato ID    Distancia L2
-----------------------------------
1               2.456789
2               3.123456
5               1.789012
7               4.567890