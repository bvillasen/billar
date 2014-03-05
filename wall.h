// #include "vector3D.h"


class Wall {
private:
  Vector3D center;
  Vector3D normal;
  int type; //0->Real, 1->Periodic
  
public:
  __device__ Wall(){}
//     normal.redefine(0.f,1.f);
//     type = 0;
//   }
  
  __device__ Wall( Vector3D c, Vector3D n, int t){
    center = c;
    normal = n;
    normal.normalize();
    type = t;
  }
  
//   __device__ ~Wall() { delete[] &center; delete[] &normal; delete[] &type; }
  
  __device__ cudaP collideTime( Vector3D &pos, Vector3D &vel ){
    if (vel*normal<=0) return -1;
    Vector3D deltaPos = center - pos;
    cudaP distNormal = deltaPos * normal;
    cudaP velNormal = vel * normal;
    return distNormal/velNormal;
  }
  
  __device__ void bounce( Vector3D &collidePos, Vector3D &vel){ //Real wall 
      cudaP factor = -2*( vel*normal );
      Vector3D deltaVel = normal/factor;
      vel = vel + deltaVel;
      vel.normalize();
  }
  __device__ void bouncePeriodic( Vector3D &collidePos, Vector3D &vel, int *region ){ //Periodic wall
    
    collidePos = collidePos - normal;
    region[0] = region[0] + normal.x;
    region[1] = region[1] + normal.y;
    region[2] = region[2] + normal.z;
  }
  __device__ bool isPeriodic(){ return type==0? false : true; }
      
      
  
};