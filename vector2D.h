#include <iostream>
#include <math.h>
using namespace std;


//Clase Vector 2D que se puede llamar desde host y devce
class Vector2D{
public:
  float x;
  float y;
  // Constructor
  __host__ __device__ Vector2D( float x0=0.f, float y0=0.f ) : x(x0), y(y0) {}
  // Destructor
//   __host__ __device__ ~Vector2D(){ delete[] &x; delete[] &y; }
  
  
  __host__ __device__ float norm( void ) { return sqrt( x*x + y*y ); };
  
  __host__ __device__ float norm2( void ) { return x*x + y*y ; };
  
  __host__ __device__ void normalize(){
    float mag = norm();
    x /= mag;
    y /= mag;
  }
  
  __host__ __device__ Vector2D operator+( Vector2D &v ){
    return Vector2D( x+v.x, y+v.y );
  }
  
  __host__ __device__ Vector2D operator-( Vector2D &v ){
    return Vector2D( x-v.x, y-v.y );
  }
  
  __host__ __device__ float operator*( Vector2D &v ){
    return x*v.x + y*v.y;
  }
  
  __host__ __device__ Vector2D operator/( float a ){
    return Vector2D( a*x, a*y );
  }  
  
  __host__ __device__ void redefine( float x0, float y0 ){
    x = x0;
    y = y0;
  }
  
  
    
  
};
  
