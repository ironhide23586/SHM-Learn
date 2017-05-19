#pragma once

#include <cuda.h>
#include <cudnn.h>
#include <cublas.h>
#include <curand.h>

#include <iostream>
#include <random>
#include <chrono>
#include <math.h>

#include "SHMatrix.h"

class Layer {
public:
  Layer();

  ~Layer();
};

