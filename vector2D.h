#include <iostream>
#include <math.h>
using namespace std;


class Vector2D{
public:
  cudaP x;
  cudaP y;
  // Constructor
  __host__ __device__ Vector2D( cudaP x0=0.f, cudaP y0=0.f ) : x(x0), y(y0) {}
  // Destructor
//   __host__ __device__ ~Vector2D(){ delete[] &x; delete[] &y; }
  
  
  __host__ __device__ cudaP norm( void ) { return sqrt( x*x + y*y ); };
  
  __host__ __device__ cudaP norm2( void ) { return x*x + y*y ; };
  
  __host__ __device__ void normalize(){
    cudaP mag = norm();
    x /= mag;
    y /= mag;
  }
  
  __host__ __device__ Vector2D operator+( Vector2D &v ){
    return Vector2D( x+v.x, y+v.y );
  }
  
  __host__ __device__ Vector2D operator-( Vector2D &v ){
    return Vector2D( x-v.x, y-v.y );
  }
  
  __host__ __device__ cudaP operator*( Vector2D &v ){
    return x*v.x + y*v.y;
  }
  
  __host__ __device__ Vector2D operator/( cudaP a ){
    return Vector2D( a*x, a*y );
  }  
  
  __host__ __device__ void redefine( cudaP x0, cudaP y0 ){
    x = x0;
    y = y0;
  }
  
  
    
  
};
  
