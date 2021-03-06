// #include "vector2D.h"

class Circle {
private:
  Vector2D center;
  cudaP radius;
  int type; //0->particles move outside, 1->particles move inside
  
public:
  __device__ Circle(){}
//     radius = 1.f;
//     type = 0;
//   }
  
  
  __device__ Circle( Vector2D c, cudaP r, int t ) : radius(r), type(t) { center = c; }
  
//   __device__ ~Circle() { delete[] &center; delete[] &radius; delete[] &type; }
  
  
  __device__ cudaP collideTime( Vector2D &pos, Vector2D &vel){
    Vector2D deltaPos = pos - center;
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
//       insideCircle = true;
//       return -1; 
//     }
    
//     return t2>0? t2 : t1;
    return t2;
  }
  
  __device__ void bounce( Vector2D &collidePos, Vector2D &vel ){
    Vector2D normal;
    normal = collidePos - center;
    normal.normalize();
    cudaP factor = -2*(vel*normal);
    normal = normal/factor;
    vel = vel + normal;
    vel.normalize();
  }
    
};