#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define TPB 64

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
        
    cudaMallocManaged((void**)&part, sizeof(particles), cudaHostAllocDefault);
    //cudaMallocManaged((void**)&param, sizeof(parameters), cudaHostAllocDefault);

    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    

    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    cudaFree(part);

    // deallocate particle variables
//   delete[] part->x;
//   delete[] part->y;
//   delete[] part->z;
//   delete[] part->u;
//   delete[] part->v;
//   delete[] part->w;
//   delete[] part->q;
}

/** particle mover */


//GPU Kernel
__global__ void mover_kernel(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x; // thread ID
    
    if (id >= part->nop) 
        return;

        // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        //for (int i=0; i <  part->nop; i++){
            xptilde = part->x[id];
            yptilde = part->y[id];
            zptilde = part->z[id];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[id] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[id] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[id] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[id] - grd->XN[ix - 1][iy][iz];
                eta[0]  =  part->y[id] -  grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[id] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[id];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[id];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[id];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[id] + qomdt2*Exl;
                vt= part->v[id] + qomdt2*Eyl;
                wt= part->w[id] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[id] = xptilde + uptilde*dto2;
                part->y[id] = yptilde + vptilde*dto2;
                part->z[id] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[id]= 2.0*uptilde - part->u[id];
            part->v[id]= 2.0*vptilde - part->v[id];
            part->w[id]= 2.0*wptilde - part->w[id];
            part->x[id] = xptilde + uptilde*dt_sub_cycling;
            part->y[id] = yptilde + vptilde*dt_sub_cycling;
            part->z[id] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[id] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[id] = part->x[id] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[id] = -part->u[id];
                    part->x[id] = 2*grd->Lx - part->x[id];
                }
            }
                                                                        
            if (part->x[id] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[id] = part->x[id] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[id] = -part->u[id];
                    part->x[id] = -part->x[id];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[id] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[id] = part->y[id] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[id] = -part->v[id];
                    part->y[id] = 2*grd->Ly - part->y[id];
                }
            }
                                                                        
            if (part->y[id] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[id] = part->y[id] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[id] = -part->v[id];
                    part->y[id] = -part->y[id];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[id] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[id] = part->z[id] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[id] = -part->w[id];
                    part->z[id] = 2*grd->Lz - part->z[id];
                }
            }
                                                                        
            if (part->z[id] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[id] = part->z[id] + grd->Lz;
                } else { // REFLECTING BC
                     part->w[id] = -part->w[id];
                    part->z[id] = -part->z[id];
                }
            }
                                                                        
            
        //}  // end of subcycling
    } // end of one particle
    
}

int mover_PC_gpu (struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    //particles* part;
    //EMfield* field;
    //grid* grd;
    //parameters* param;

    //cudaMalloc(&part, sizeof(particles));
    cudaMallocManaged(&field, sizeof(EMfield));
    cudaMallocManaged(&grd, sizeof(grid));
    //cudaMalloc(&param, sizeof(parameters));

    ////cudaMemcpy(part, part, sizeof(particles),cudaMemcpyHostToDevice);
    ////cudaMemcpy(field, field, sizeof(EMfield),cudaMemcpyHostToDevice);
    ////cudaMemcpy(grd, grd, sizeof(grid),cudaMemcpyHostToDevice);
    ////cudaMemcpy(param, param, sizeof(parameters),cudaMemcpyHostToDevice);


    int gridnum = (part->nop+TPB-1)/TPB;
    int blocksize = TPB;

    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
    mover_kernel<<<gridnum, blocksize>>>(part, field, grd, param);
    cudaDeviceSynchronize();

    //cudaMemcpy(part, part, sizeof(particles), cudaMemcpyDeviceToHost);
    //cudaMemcpy(field, field, sizeof(EMfield), cudaMemcpyDeviceToHost);
    //cudaMemcpy(grd, grd, sizeof(grid), cudaMemcpyDeviceToHost);
    //cudaMemcpy(param, param, sizeof(parameters), cudaMemcpyDeviceToHost);

    //cudaFree(part);
    cudaFree(field);
    cudaFree(grd);
    //cudaFree(param);

    return 0;
}

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}
