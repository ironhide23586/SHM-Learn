#include <stdlib.h>
#include <iostream>

using namespace std;

int my_ceilf_division(float a, float b) {
  return 1 + ((a - 1) / b);
}

int main() {
   int a = 5;
   int b = 2;
   int c = my_ceilf_division(a, b);
   std::cout << c << std::endl;
}
