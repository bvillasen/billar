// #include "vector3D.h"

class Sphere {
private:
  Vector3D center;
  cudaP radius;
  int type; //0->particles move outside, 1->particles move inside
  
public:
  __device__ Sphere(){}
//     radius = 1.f;
//     type = 0;
//   }
  
  
  __device__ Sphere( Vector3D c, cudaP r, int t ) : radius(r), type(t) { center = c; }
  
//   __device__ ~Sphere() { delete[] &center; delete[] &radius; delete[] &type; }
  
  
  __device__ cudaP collideTime( Vector3D &pos, Vector3D &vel){
    Vector3D deltaPos = pos - center;
    cudaP B = vel * deltaPos;
    cudaP C = deltaPos*deltaPos - radius*radius;
    cudaP d = B*B - C;
    if ( d<0 ) return -1.f; //particle doesnt collide with circle 
//     if (deltaPos.norm() < radius*0.5 ){ restart = true; return -1.f; }
//     if ( abs(deltaPos.norm()-radius) < 0.000001) return -1;
    
    cudaP t1 = -B + sqrt(d);
    cudaP t2 = -B - sqrt(d);
    
//     if (t1>=0 and t2<=0) return t1;
    
//     if ( (deltaPos.norm()-radius)<-0){
//       insideSphere = true;
//       return -1; 
//     }
    
//     return t2>0? t2 : t1;
    return t2;
  }
  
  __device__ void bounce( Vector3D &collidePos, Vector3D &vel ){
    Vector3D normal;
    normal = collidePos - center;
    normal.normalize();
    cudaP factor = -2*(vel*normal);
    normal = normal/factor;
    vel = vel + normal;
    vel.normalize();
  }
    
};