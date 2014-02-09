// #include "vector2D.h"

class Circle {
private:
  Vector2D center;
  float radius;
  int type; //0->particles move outside, 1->particles move inside
  
public:
  __device__ Circle(){}
//     radius = 1.f;
//     type = 0;
//   }
  
  
  __device__ Circle( Vector2D c, float r, int t ) : radius(r), type(t) { center = c; }
  
//   __device__ ~Circle() { delete[] &center; delete[] &radius; delete[] &type; }
  
  
  __device__ float collideTime( Vector2D &pos, Vector2D &vel){
    Vector2D deltaPos = pos - center;
    float B = vel * deltaPos;
    float C = deltaPos*deltaPos - radius*radius;
    float d = B*B - C;
    if ( d<0 ) return -1.f; //particle doesnt collide with circle 
//     if (deltaPos.norm() < radius*0.5 ){ restart = true; return -1.f; }
//     if ( abs(deltaPos.norm()-radius) < 0.000001) return -1;
    
    float t1 = -B + sqrt(d);
    float t2 = -B - sqrt(d);
    
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
    float factor = -2*(vel*normal);
    normal = normal/factor;
    vel = vel + normal;
    vel.normalize();
  }
    
};