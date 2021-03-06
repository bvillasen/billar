// #include "vector2D.h"


class Line {
private:
  Vector2D center;
  Vector2D normal;
  int type; //0->Real, 1->Periodic
  
public:
  __device__ Line(){}
//     normal.redefine(0.f,1.f);
//     type = 0;
//   }
  
  __device__ Line( Vector2D c, Vector2D n, int t){
    center = c;
    normal = n;
    normal.normalize();
    type = t;
  }
  
//   __device__ ~Line() { delete[] &center; delete[] &normal; delete[] &type; }
  
  __device__ cudaP collideTime( Vector2D &pos, Vector2D &vel ){
    if (vel*normal<=0) return -1;
    Vector2D deltaPos = center - pos;
    cudaP distNormal = deltaPos * normal;
    cudaP velNormal = vel * normal;
    return distNormal/velNormal;
  }
  
  __device__ void bounce( Vector2D &collidePos, Vector2D &vel){ //Real wall 
      cudaP factor = -2*( vel*normal );
      Vector2D deltaVel = normal/factor;
      vel = vel + deltaVel;
      vel.normalize();
  }
  __device__ void bouncePeriodic( Vector2D &collidePos, Vector2D &vel, int *region ){ //Periodic wall
    
    collidePos = collidePos - normal;
    region[0] = region[0] + normal.x;
    region[1] = region[1] + normal.y;
  }
  __device__ bool isPeriodic(){ return type==0? false : true; }
      
      
  
};
  