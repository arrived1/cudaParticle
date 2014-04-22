#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <GL/glew.h>
#include <GL/glut.h>

#include <vector>
#include <utility>
#include <cmath>
#include <string>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cufft.h>
#include <cuda.h>

/*
#include "particleSystem.h"
#include "emiter.h"

const float a = 10.f;
const float dt = 0.001f;
const float pointSize = 1.f;

const float v = 16.f;
const float radius = 0.02f;




//unsigned step = 0;
ParticleSystem<float2> pSystem(40);
Emiter<ParticleSystem<float2>, float2> emiter(pSystem, make_float2(-5.f, 0.f));

bool collision(float2 p1, float2 p2);
void checkState(unsigned i);

void renderDots() {
    glPointSize(pointSize);
    glBegin(GL_POINTS);
    
    float3 color;
    for (unsigned i = 0; i < pSystem.getParticleAmount(); i++) {
        color = pSystem.getParticleColor(i);
        glColor3f(color.x, color.y, color.z);
        glVertex3f(pSystem.getParticlePos(i).x, pSystem.getParticlePos(i).y, 0.0f);
    }
    glEnd();
}

void renderEmiter() {
    glPointSize(10.f);
    float2 pos = emiter.getPosition();

    glBegin(GL_POINTS);
        glColor3f(1.f, 1.f, 1.f);    
        glVertex3f(pos.x, pos.y, 0.0f);
    glEnd();
}

void renderLines() {
    glBegin(GL_LINES);
    glColor3f(0.f, 1.f, 0.f);
        glVertex2f(-4.f, 10.f);
        glVertex2f(-4.f, -10.f);

        glVertex2f(4.8f, 10.f);
        glVertex2f(4.8f, -10.f);
    glEnd();
}

void display(void) {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Clear the background of our window to red  
    glClear(GL_COLOR_BUFFER_BIT); //Clear the colour buffer (more buffers later on)  
    glLoadIdentity(); // Load the Identity Matrix to reset our drawing locations  

    glTranslatef(0.0f, 0.0f, -(a+2)); // Push eveything 5 units back into the scene, otherwise we won't see the primitive  

    //renderPrimitive(); // Render the primitive
    
    renderDots();
    renderEmiter();
    renderLines();
    glFlush(); // Flush the OpenGL buffers to the window  
}

void dupar_function(int value) {
    for (unsigned i = 0; i < pSystem.getParticleAmount(); i++)
        emiter.emit(i);

    std::cout << "Startuje synulacje" << std::endl;
    for (unsigned i = 0; i < pSystem.getParticleAmount(); i++) 
        for (unsigned j = 0; j < pSystem.getParticleAmount(); j++) {
            
            if(i == j)
                continue;

            checkState(i);            

            if (pSystem.getParticleState(i) == inBox) {
                if (collision(pSystem.getParticlePos(i), pSystem.getParticlePos(j))) {
                    std::cout << "kolizja!!! " << i << " z " << j << std::endl;
                    float2 vel = pSystem.getParticleVel(i);
                    vel.x *= -1;
                    vel.y *= -1;
                    pSystem.setParticleVel(i, vel);

                    pSystem.getNextParticlePos(i) = pSystem.getPrevParticlePos(i);
                }
            }
        }
    


    for (unsigned i = 0; i < pSystem.getParticleAmount(); i++) {
        pSystem.prepareMove(i, dt);
        pSystem.move(i);
    }

    pSystem.incrementStep();


    glutPostRedisplay();
    glutduparFunc(200, dupar_function, 0);
}

bool collision(float2 p1, float2 p2) {
    float d_kwadrat = (p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y);
    float r_kwadrat = (2 * radius) * (2 * radius);

    if (d_kwadrat <= r_kwadrat) 
        return true;

    return false;
}

void checkState(unsigned i) {
    int x = pSystem.getParticlePos(i).x;
    
    if (x > -4.f)
        pSystem.setParticleState(i, inBox);
    
    if (x > 4.8f) {
        emiter.backToEmiter(i);
    }   
}

void reshape(int width, int height) {
    
    glViewport(0, 0, (GLsizei)width, (GLsizei)height); // Set our viewport to the size of our window  
    glMatrixMode(GL_PROJECTION); // Switch to the projection matrix so that we can manipulate how our scene is viewed  
    glLoadIdentity(); // Reset the projection matrix to the identity matrix so that we don't get any artifacts (cleaning up)  
    gluPerspective(60, (GLfloat)width / (GLfloat)height, 1.0, 100.0); // Set the Field of view angle (in degrees), the aspect ratio of our window, and the new and far planes  
    glMatrixMode(GL_MODELVIEW); // Switch back to the model view matrix, so that we can start drawing shapes correctly  
}


int main(int argc, char **argv) {
    //initParticle();

    //std::cout << particles << std::endl;
    glutInit(&argc, argv); // Initialize GLUT  
    glutInitDisplayMode(GLUT_SINGLE); // Set up a basic display buffer (only single buffered for now)  
    glutInitWindowSize(500, 500); // Set the width and height of the window  
    glutInitWindowPosition(500, 100); // Set the position of the window  
    glutCreateWindow("Your first OpenGL Window"); // Set the title for the window  

    glutDisplayFunc(display); // Tell GLUT to use the method "display" for rendering  
    glutduparFunc(1, dupar_function, 0);

    glutReshapeFunc(reshape); // Tell GLUT to use the method "reshape" for reshaping  

    glutMainLoop(); // Enter GLUT's main loop  
}

*/



#define _USE_MATH_DEFINES
#include <iostream>
#include <math.h>
//#include <cdupa>
#include <vector_types.h>

//Constants
const unsigned int window_width = 768;
const unsigned int window_height = 768;

const unsigned int mesh_width = 1024;
const unsigned int mesh_height = 1024;

float rnd1[mesh_width*mesh_height];
float rnd2[mesh_width*mesh_height];

//Mouse controls
int mouse_x, mouse_y;
int buttons = 0;
float translate_z = -3.0;

//VBO variables
GLuint vbo;
void *d_vbo_buffer = NULL;

float dupa = 0.0;

//Device pointers
float4 *d_array;
float *d_rnd1, *d_rnd2;

void keyboard(unsigned char key, int, int)
{
    switch(key) {
    case(27) :
        exit(0);
        break;
    case('a'):
        if (buttons!=10)
            buttons=10;
        else
            buttons=0;
        break;
    }
}

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        buttons = 0;
    }

    mouse_x = x;
    mouse_y = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - mouse_x;
    dy = y - mouse_y;

    if (buttons & 4)
        translate_z += dy * 0.01;

    mouse_x = x;
    mouse_y = y;
}

union Color
{
    float c;
    uchar4 components;
};

__global__ void initialize_kernel(float4* pos, unsigned int width, unsigned int height, float dupa, float4* vel, float* rnd1, float* rnd2)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    //Calculate the initial coordinates
    float u = x / (float) width + rnd1[y*width+x];
    float v = y / (float) height + rnd2[y*width+x];

    //Calculate a simple sine wave pattern
    float freq = 2.0f;
    float w = sinf(u*freq + dupa) * cosf(v*freq + dupa) * 0.2f;

    //Set the initial color
    Color temp;
    temp.components = make_uchar4(0,255,255,1);

    //Set initial position, color and velocity
    pos[y*width+x] = make_float4(u, w, v, temp.c);
    vel[y*width+x] = make_float4(0.0, 0.0, 0.0, 1.0f);
}

__global__ void particles_kernel(float4* pos, unsigned int width, unsigned int height, float dupa, float X, float Y, float4* vel, int buttons)
{
    const float speed = 0.0005f;
    const float threshold = 0.1f;
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float u = x / (float) width;
    float v = y / (float) height;

    float xX = (X - width/2 + 128)/(float)width*4.5f;
    float yY = (Y - height/2 + 128)/(float)height*4.5f;
    float dx = -pos[y*width+x].x + xX;
    float dz = -pos[y*width+x].z + yY;
    float length = sqrtf(dx*dx+dz*dz);
    if (buttons==10)
    {
        vel[y*width+x].x=0;
        vel[y*width+x].z=0;
        dx = -pos[y*width+x].x + u;
        dz = -pos[y*width+x].z + v;
        length = sqrtf(dx*dx+dz*dz);
        pos[y*width+x].x+=dx/length*speed*10;
        pos[y*width+x].z+=dz/length*speed*10;
    }
    else if (!(buttons & 4) && !(buttons & 6))
    {
        float2 normalized = make_float2(dx/length*speed, dz/length*speed);
        vel[y*width+x].x+=normalized.x;
        vel[y*width+x].z+=normalized.y;
        dx = vel[y*width+x].x;
        dz = vel[y*width+x].z;
        float velocity = sqrtf(dx*dx+dz*dz);
        if (velocity>threshold)
        {
            vel[y*width+x].x=dx/velocity*threshold;
            vel[y*width+x].z=dz/velocity*threshold;
        }
        Color temp;
        temp.components = make_uchar4(128/length,(int)(255/(velocity*51)),255,10);
        if (pos[y*width+x].x<-5.0f && vel[y*width+x].x<0.0)
            vel[y*width+x].x=-vel[y*width+x].x;
        if (pos[y*width+x].x>5.0f && vel[y*width+x].x>0.0)
            vel[y*width+x].x=-vel[y*width+x].x;
        pos[y*width+x].x+=vel[y*width+x].x;
        pos[y*width+x].z+=vel[y*width+x].z;
        pos[y*width+x].w = temp.c;
    }
    else if (!(buttons & 4))
    {
        vel[y*width+x].x=0;
        vel[y*width+x].z=0;
        pos[y*width+x].x+=dx/length*speed*10;
        pos[y*width+x].z+=dz/length*speed*10;
        Color temp;
        temp.components = make_uchar4(255/length,255/length, 255, 10);
        pos[y*width+x].w = temp.c;
    }

    float freq = 2.0f;
    float w = sinf(u*freq + dupa) * cosf(v*freq + dupa) * 0.2f;

    pos[y*width+x].y=w;
}

void particles(GLuint vbo)
{
    //Map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    cudaGLMapBufferObject((void**)&dptr, vbo);

    //Run the particles kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    particles_kernel<<< grid, block>>>(dptr, mesh_width, mesh_height, dupa, mouse_x, mouse_y, d_array, buttons);

    //Unmap buffer object
    cudaGLUnmapBufferObject(vbo);
}

void initialize(GLuint vbo)
{
    //Map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    cudaGLMapBufferObject((void**)&dptr, vbo);

    //Run the initialization kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    initialize_kernel<<< grid, block>>>(dptr, mesh_width, mesh_height, dupa, d_array, d_rnd1, d_rnd2);

    //Unmap buffer object
    cudaGLUnmapBufferObject(vbo);
}

static void display(void)
{
    //Process particles using CUDA kernel
    particles(vbo);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //View matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(90.0, 1.0, 0.0, 0.0);

    //Render from VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(3, GL_FLOAT, 16, 0);
    glColorPointer(4,GL_UNSIGNED_BYTE,16,(GLvoid*)12);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
    glutPostRedisplay();

    dupa += 0.01;
}

void createVBO(GLuint* vbo)
{
    //Create vertex buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    //Initialize VBO
    unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //Register VBO with CUDA
    cudaGLRegisterBufferObject(*vbo);
}

void initGL(int argc, char **argv)
{
    glutInit(&argc, argv);

    //Setup window
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Million particles");

    //Register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    //GLEW initialization
    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 ")) {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        exit(0);
    }

    //Clear
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    //Viewport
    glViewport(0, 0, window_width, window_height);

    //Projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);
}

int main(int argc, char** argv)
{
    initGL(argc, argv);

    //Create VBO
    createVBO(&vbo);

    //Initialize random arrays
    for (int i=0;i<mesh_height*mesh_width;++i)
        rnd1[i]=(rand()%100-100)/2000.0f;
    for (int i=0;i<mesh_height*mesh_width;++i)
        rnd2[i]=(rand()%100-100)/2000.0f;

    //CUDA allocation and copying
    cudaMalloc(&d_array, mesh_width*mesh_height*sizeof(float4));
    cudaMalloc(&d_rnd1, mesh_width*mesh_height*sizeof(float));
    cudaMemcpy(d_rnd1, rnd1, mesh_height*mesh_width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_rnd2, mesh_width*mesh_height*sizeof(float));
    cudaMemcpy(d_rnd2, rnd2, mesh_height*mesh_width*sizeof(float), cudaMemcpyHostToDevice);

    initialize(vbo);

    glutMainLoop();

    //Free CUDA variables
    cudaFree(d_array);
    cudaFree(d_rnd1);
    cudaFree(d_rnd2);
    return 0;
}