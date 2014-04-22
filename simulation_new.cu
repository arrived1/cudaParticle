#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <GL/glew.h>
#include <GL/glut.h>

#include <vector>
#include <utility>
#include <cmath>
#include <string>
#include <time.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cufft.h>
#include <cuda.h>
#include <helper_functions.h>


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

float dt = 0.0f;

//Device pointers
float4 *d_vel;
float *d_rnd1, *d_rnd2;

//FPS
int fpsCount = 0;      // FPS count for averaging
int fpsLimit = 8;      // FPS limit for sampling
unsigned int frameCount = 0;
StopWatchInterface *timer = NULL;


void keyboard(unsigned char key, int, int);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void initGL(int argc, char **argv);
void display(void);
void computeFPS();


union Color
{
    float c;
    uchar4 components;
};

__global__ void initialize_kernel(float4* pos, unsigned int width, unsigned int height, float dt, 
                                  float4* vel, float* rnd1, float* rnd2)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    //Calculate the initial coordinates
    float u = x / (float) width + rnd1[y*width+x];
    float v = y / (float) height + rnd2[y*width+x];

    //Calculate a simple sine wave pattern
    float freq = 2.0f;
    float w = sinf(u*freq + dt) * cosf(v*freq + dt) * 0.2f;

    //Set the initial color
    Color temp;
    temp.components = make_uchar4(0,255,255,1);

    //Set initial position, color and velocity
    pos[y*width+x] = make_float4(u, w, v, temp.c);
    vel[y*width+x] = make_float4(0.0, 0.0, 0.0, 1.0f);
}

__global__ void particles_kernel(float4* pos, unsigned int width, unsigned int height, float dt, 
                                float X, float Y, float4* vel, int buttons)
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
    float w = sinf(u*freq + dt) * cosf(v*freq + dt) * 0.2f;

    pos[y*width+x].y=w;
}

void particles(GLuint vbo)
{
    //Map OpenGL buffer object for writing from CUDA
    float4 *d_pos;
    cudaGLMapBufferObject((void**)&d_pos, vbo);

    //Run the particles kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    particles_kernel<<< grid, block>>>(d_pos, mesh_width, mesh_height, dt, mouse_x, mouse_y, d_vel, buttons);

    //Unmap buffer object
    cudaGLUnmapBufferObject(vbo);
}

void initialize(GLuint vbo)
{
    //Map OpenGL buffer object for writing from CUDA
    float4 *d_pos;
    cudaGLMapBufferObject((void**)&d_pos, vbo);

    //Run the initialization kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    initialize_kernel<<< grid, block>>>(d_pos, mesh_width, mesh_height, dt, d_vel, d_rnd1, d_rnd2);

    //Unmap buffer object
    cudaGLUnmapBufferObject(vbo);
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


int main(int argc, char** argv)
{
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    initGL(argc, argv);

    //Create VBO
    createVBO(&vbo);

    //Initialize random arrays
    for (int i = 0; i < mesh_height * mesh_width; ++i)
        rnd1[i] = (rand() % 100 - 100) / 2000.0f;
    for (int i = 0; i < mesh_height * mesh_width; ++i)
        rnd2[i] = (rand() % 100 - 100) / 2000.0f;

    //CUDA allocation and copying
    cudaMalloc(&d_vel, mesh_width * mesh_height * sizeof(float4));
    cudaMalloc(&d_rnd1, mesh_width * mesh_height * sizeof(float));
    cudaMemcpy(d_rnd1, rnd1, mesh_height * mesh_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_rnd2, mesh_width * mesh_height * sizeof(float));
    cudaMemcpy(d_rnd2, rnd2, mesh_height * mesh_width * sizeof(float), cudaMemcpyHostToDevice);

    initialize(vbo);

    glutMainLoop();

    //Free CUDA variables
    cudaFree(d_vel);
    cudaFree(d_rnd1);
    cudaFree(d_rnd2);
    return 0;
}



/////////////////////////////////////// HELPERS /////////////////////////
#define MAX(a,b) ((a > b) ? a : b)

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "simulation (%s): %3.1f fps","", ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = (int)MAX(ifps, 1.f);
        sdkResetTimer(&timer);
    }
}


/////////////////////////////////////// OPENGL //////////////////////////

void display(void)
{
    sdkStartTimer(&timer);

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

    dt += 0.01;

    sdkStopTimer(&timer);
    computeFPS();
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
    //float dx = x - mouse_x;
    float dy = y - mouse_y;

    if (buttons & 4)
        translate_z += dy * 0.01;

    mouse_x = x;
    mouse_y = y;
}